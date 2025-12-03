# ===================================================================
# 690m ICT SMART MONEY BOT - Gate.io Futures (Perpetual Swap)
# - Lucrează cu perechea BTC/USDT:USDT (sau alte perp-uri Gate.io)
# - Concepte ICT: 690m bias, MSS 1h, FVG 4m, Kill Zone
# - Management de risc pe baza balanței USDT
# - EXECUTĂ doar ordinul MARKET; SL/TP sunt calculate și afișate
# ===================================================================

import time
from datetime import datetime

import ccxt
import pandas as pd
import pytz

# ==================== CONFIGURARE - MODIFICĂ AICI ====================

API_KEY = "42bb2de8ebdc7c3db84995e203e6752f"        # <<< cheia ta Gate.io
API_SECRET = "709f16ca0e3da2c8f946f80d336180621ac774aa9aad374a848ddb822975de1a"      # <<< secretul tău Gate.io

USE_TESTNET = False                  # True = sandbox.gateio.ws, False = live

SYMBOL = "XRP/USDT:USDT"             # exemplu perp BTC marginit în USDT
MARGIN_ASSET = "USDT"                # activul cu care este marginită perechea

LEVERAGE = 5                         # levier (setează-l la fel și în UI Gate.io)
RISK_PERCENT = 0.5                   # risc per trade (% din balanța USDT în futures)
MAX_POSITIONS = 1                    # număr maxim de poziții simultane

TIMEZONE = "Europe/Bucharest"        # fusul orar pentru Kill Zone și mesaje

# ==================== CONEXIUNE GATE.IO FUTURES (SWAP) ====================

if USE_TESTNET:
    # Gate.io sandbox (testnet)
    exchange = ccxt.gateio({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",       # perpetual futures
        },
    })
    exchange.set_sandbox_mode(True)
else:
    exchange = ccxt.gateio({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",       # perpetual futures
        },
    })

# ==================== FUNCȚII OHLCV ====================

def get_ohlcv(symbol, timeframe, limit=200):
    """Preia OHLCV de la Gate.io Swap și îl pune într-un DataFrame."""
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # conversie în fusul tău orar (doar pentru afișare)
        try:
            tz = pytz.timezone(TIMEZONE)
            df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
        except Exception:
            pass

        df.set_index("timestamp", inplace=True)
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        return df
    except Exception as e:
        print(f"❌ Eroare date ({timeframe}): {e}")
        return None


def get_ohlcv_690m(symbol, limit_bars=10):
    """
    Simulează timeframe-ul 690m (11.5h) prin resampling din 15m.
    """
    base_tf = "15m"
    need_15m = limit_bars * int(690 / 15) + 20
    df_15m = get_ohlcv(symbol, base_tf, need_15m)
    if df_15m is None or df_15m.empty:
        return None

    # resample trebuie în UTC
    df_15m_resample = df_15m.copy()
    df_15m_resample.index = df_15m_resample.index.tz_convert("UTC")

    df_690m = df_15m_resample.resample("690min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    df_690m = df_690m.tail(limit_bars)
    return df_690m if not df_690m.empty else None

# ==================== CONDIȚII ICT (Bias, MSS, FVG, Kill Zone) ====================

def detect_fvg(df, bullish=True):
    """Detectează un Fair Value Gap simplificat pe ultimele 3 lumânări."""
    if df is None or len(df) < 3:
        return False

    if bullish:
        # Low(-3) mai mare decât High(-1) => gol în jos (bullish FVG)
        return df["Low"].iloc[-3] > df["High"].iloc[-1]
    else:
        # High(-3) mai mic decât Low(-1) => gol în sus (bearish FVG)
        return df["High"].iloc[-3] < df["Low"].iloc[-1]


def get_690m_bias():
    """Determină direcția pe timeframe-ul 690m (11.5h)."""
    df = get_ohlcv_690m(SYMBOL, 10)
    if df is None or len(df) < 3:
        return "RANGING"

    last = df.iloc[-2]   # ultima bară închisă
    prev = df.iloc[-3]

    # Bullish: bară verde + liquidity sweep jos
    if last["Close"] > last["Open"] and last["Low"] < prev["Low"]:
        return "BULLISH"
    # Bearish: bară roșie + liquidity sweep sus
    elif last["Close"] < last["Open"] and last["High"] > prev["High"]:
        return "BEARISH"

    return "RANGING"


def detect_mss():
    """Market Structure Shift pe 1h (ruperea unui swing high/low)."""
    df = get_ohlcv(SYMBOL, "1h", 60)
    if df is None or len(df) < 20:
        return False

    price = df["Close"].iloc[-1]
    swing_high = df["High"].iloc[-10:-2].max()
    swing_low = df["Low"].iloc[-10:-2].min()

    return price > swing_high or price < swing_low


def is_kill_zone():
    """London (9-12) sau NY (14-17) Kill Zone în fusul orar setat."""
    hour = datetime.now(pytz.timezone(TIMEZONE)).hour
    return (9 <= hour <= 12) or (14 <= hour <= 17)

# ==================== RISC & POZIȚII ====================

def calculate_position_size(balance, risk_pct, entry, sl):
    """
    Calculează mărimea poziției (cantitate de asset, ex: BTC pentru BTC/USDT:USDT) în funcție de risc.
    """
    price_diff = abs(entry - sl)
    if price_diff == 0:
        return 0.0

    risk_amount = balance * (risk_pct / 100.0)
    # qty ≈ (risk_amount * leverage) / (price_diff * entry)
    qty = (risk_amount * LEVERAGE) / (price_diff * entry)
    return float(round(qty, 6))  # avem nevoie de 6 zecimale pe crypto


def get_open_positions_count():
    """Numărul de poziții active (non-zero) pe acest simbol în Gate.io Swap."""
    try:
        positions = exchange.fetch_positions([SYMBOL])
        active = [
            p for p in positions
            if float(p.get("contracts") or p.get("amount") or 0) != 0
        ]
        return len(active)
    except Exception as e:
        print(f"⚠️ Eroare la citirea pozițiilor: {e}")
        return 0

# ==================== ORDERS (entry; SL/TP doar calculate) ====================

def execute_trade(direction, entry, sl, tp):
    """
    Calculează mărimea și lansează trade-ul cu ordin MARKET pe Gate.io Swap.
    SL și TP sunt calculate și afișate, dar NU se trimit ca ordine condiționale
    (poți adăuga ulterior create_order cu tipul corect pentru Gate.io).
    """
    try:
        balance = exchange.fetch_balance()
        futures_balance = balance.get(MARGIN_ASSET, {}) or balance.get(f"{MARGIN_ASSET}:{MARGIN_ASSET}", {})
        free_balance = float(futures_balance.get("free", 0))

        if free_balance <= 0:
            print(f"⚠️ Balanță {MARGIN_ASSET} insuficientă.")
            return False

        qty = calculate_position_size(free_balance, RISK_PERCENT, entry, sl)
        if qty <= 0:
            print("⚠️ Cantitate invalidă (qty <= 0).")
            return False

        side = "buy" if direction.upper() == "BUY" else "sell"

        print(
            f"📐 Position size: balance={free_balance:.2f} {MARGIN_ASSET}, "
            f"qty={qty}, entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}"
        )
        print(f"📥 Trimit ordin MARKET {side.upper()} pe {SYMBOL} ...")

        order = exchange.create_order(
            symbol=SYMBOL,
            type="market",
            side=side,
            amount=qty
        )
        print(f"✅ Ordin MARKET executat: {order.get('id', 'fără id')}")

        print("⚠️ NOTE: SL/TP NU au fost trimise ca ordine pe Gate.io din cod.")
        print("          Setează manual sau adaugă logică suplimentară pentru ordine condiționale.")

        return True

    except Exception as e:
        print(f"❌ Eroare trade: {e}")
        return False

# ==================== LOOP PRINCIPAL STRATEGIE ICT 690m ====================

def ict_690m_bot():
    """Loopul principal al botului ICT 690m pe Gate.io Futures (swap)."""
    print("=" * 70)
    print("🚀 690m ICT SMART MONEY BOT - Gate.io Futures (swap)")
    print("=" * 70)

    while True:
        try:
            now = datetime.now(pytz.timezone(TIMEZONE))
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] 🔍 Scan piață pentru {SYMBOL}...")

            # 1. Poziții existente
            open_pos = get_open_positions_count()
            print(f"📌 Poziții deschise: {open_pos}")
            if open_pos >= MAX_POSITIONS:
                print("⏸️ Există deja poziție activă, aștept 5 minute...")
                time.sleep(300)
                continue

            # 2. Bias 690m
            bias = get_690m_bias()
            print(f"📊 Bias 690m: {bias}")
            if bias == "RANGING":
                print("🔁 Piață în range, nu forțez intrări. Revin peste 5 minute.")
                time.sleep(300)
                continue

            # 3. Kill Zone
            if not is_kill_zone():
                print("⏰ Nu este Kill Zone (9-12 sau 14-17). Revin peste 10 minute.")
                time.sleep(600)
                continue

            # 4. MSS pe 1h
            if not detect_mss():
                print("⏳ Aștept MSS pe 1h pentru confirmare. Revin peste 4 minute.")
                time.sleep(240)
                continue

            print("✅ MSS confirmat pe 1h!")

            # 5. Entry pe 4 minute
            df_4m = get_ohlcv(SYMBOL, "5m", 100)
            if df_4m is None or len(df_4m) < 20:
                print("⚠️ Date insuficiente pe 4m. Revin peste 1 minut.")
                time.sleep(60)
                continue

            price = float(df_4m["Close"].iloc[-1])

            # "pip" relativ, ~0.05% din preț (buffer pentru SL)
            pip = price * 0.0005

            # BUY setup
            if bias == "BULLISH" and detect_fvg(df_4m, bullish=True):
                print("🟢 FVG BULLISH detectat pe 4m!")
                recent_lows = df_4m["Low"].iloc[-10:]
                sl = float(recent_lows.min() - 5 * pip)
                tp = float(price + (price - sl) * 3.5)
                execute_trade("BUY", price, sl, tp)
                print("😴 Pauză 10 minute după BUY.")
                time.sleep(600)

            # SELL setup
            elif bias == "BEARISH" and detect_fvg(df_4m, bullish=False):
                print("🔴 FVG BEARISH detectat pe 4m!")
                recent_highs = df_4m["High"].iloc[-10:]
                sl = float(recent_highs.max() + 5 * pip)
                tp = float(price - (sl - price) * 3.5)
                execute_trade("SELL", price, sl, tp)
                print("😴 Pauză 10 minute după SELL.")
                time.sleep(600)

            else:
                print("⚖️ Niciun setup valid FVG în direcția bias-ului. Revin peste 4 minute.")
                time.sleep(240)

        except KeyboardInterrupt:
            print("\n⛔ Bot oprit manual (KeyboardInterrupt).")
            break
        except Exception as e:
            print(f"❌ Eroare în loop: {e}")
            print("⏳ Reîncerc în 60 de secunde...")
            time.sleep(60)

# ==================== MAIN ====================

if __name__ == "__main__":
    try:
        print("🔌 Conectare la Gate.io Futures (swap)...")
        markets = exchange.load_markets()
        if SYMBOL not in markets:
            raise ValueError(
                f"Simbolul {SYMBOL} nu există în markets Gate.io. "
                f"Verifică denumirea exactă din fereastra de futures."
            )

        print(f"✅ Conectat la Gate.io ({'TESTNET' if USE_TESTNET else 'LIVE'})")
        print(f"📈 Symbol: {SYMBOL}")

        # Încearcă să setezi levierul dorit (dacă API-ul permite)
        try:
            exchange.set_leverage(LEVERAGE, SYMBOL)
            print(f"⚙️ Levier setat la {LEVERAGE}x pentru {SYMBOL} (dacă era permis).")
        except Exception as e:
            print(f"⚠️ Nu am putut seta levierul prin API: {e}")
            print("   👉 Asigură-te că levierul este setat manual în interfața Gate.io.")

        ict_690m_bot()

    except Exception as e:
        print(f"❌ Eroare inițializare: {e}")
        print("\n💡 Verifică API Key, Secret, activarea Futures și corectitudinea simbolului.")
