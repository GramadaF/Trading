# ===================================================================
# 690m ICT SMART MONEY BOT - Gate.io Futures (Perpetual Swap)
# TUNED FOR CRYPTO BEHAVIOR + SCALPER MODULE
#
# - Exchange: Gate.io (swap / perpetual futures)
# - Symbol example: BTC/USDT:USDT
# - ICT swing:
#     * 690m higher-timeframe bias (resampled from 15m)
#     * MSS (Market Structure Shift) on 1h
#     * FVG (Fair Value Gap) on 5m
# - SCALPER:
#     * Folosește același HTF bias + MSS + Kill Zone
#     * Intrare pe FVG pe timeframe mai mic (1m by default)
# - CRYPTO TUNING:
#     * Crypto Kill Zones (Asia + London + New York)
#     * Weekend filter (no trades Sat/Sun)
#     * ATR-based volatility filter
# ===================================================================

import time
from datetime import datetime

import ccxt
import pandas as pd
import pytz

# ==================== CONFIG - MODIFY HERE ====================

API_KEY = "42bb2de8ebdc7c3db84995e203e6752f"        # <<< Gate.io API key
API_SECRET = "709f16ca0e3da2c8f946f80d336180621ac774aa9aad374a848ddb822975de1a"      # <<< Gate.io Secret

USE_TESTNET = False                  # True = sandbox.gateio.ws, False = live

SYMBOL = "BTC/USDT:USDT"             # Example perp: BTC/USDT:USDT
MARGIN_ASSET = "USDT"                # Margin asset

LEVERAGE = 5                         # Match this with Gate.io UI
RISK_PERCENT = 0.5                   # % din balanță pe trade swing
MAX_POSITIONS = 1                    # Max simultaneous positions

TIMEZONE = "Europe/Bucharest"        # Your timezone (used for Kill Zones)

# --- CRYPTO TUNING ---

ENABLE_WEEKEND_FILTER = True         # Skip trading on Sat/Sun
ENABLE_VOLATILITY_FILTER = True      # Use ATR filter on 5m

# ATR settings pentru swing (5m)
ATR_PERIOD = 20                      # Period for ATR on 5m
ATR_MIN_MULT = 0.001                 # Min ATR = 0.1% of price
ATR_MAX_MULT = 0.03                  # Max ATR = 3% of price

# --- SCALPER MODULE ---

SCALPER_ENABLED = True               # Activează / dezactivează scalper
SCALPER_TIMEFRAME = "1m"             # ex: "1m" sau "3m"
SCALPER_ATR_PERIOD = 14              # ATR pentru scalper
SCALPER_RISK_PERCENT = 0.25          # risc per trade pentru scalper (mai mic)
SCALPER_R_MULT = 2.0                 # Risk:Reward pentru scalper

# ==================== GATE.IO CONNECTION ====================

if USE_TESTNET:
    exchange = ccxt.gateio({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",   # perpetual futures
        },
    })
    exchange.set_sandbox_mode(True)
else:
    exchange = ccxt.gateio({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",   # perpetual futures
        },
    })

# ==================== OHLCV HELPERS ====================

def get_ohlcv(symbol, timeframe, limit=200):
    """Fetch OHLCV from Gate.io Swap -> pandas DataFrame."""
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # convert to local timezone only for display / logic
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
    Build synthetic 690m (11.5h) candles from 15m data (crypto-friendly).
    Uses '690min' instead of '690T' to avoid pandas FutureWarning.
    """
    base_tf = "15m"
    need_15m = limit_bars * int(690 / 15) + 20
    df_15m = get_ohlcv(symbol, base_tf, need_15m)
    if df_15m is None or df_15m.empty:
        return None

    # Resample in UTC for consistency
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

# ==================== ICT CONDITIONS ====================

def detect_fvg(df, bullish=True):
    """Simple Fair Value Gap detection on last 3 candles."""
    if df is None or len(df) < 3:
        return False

    if bullish:
        # Bullish FVG: low[-3] > high[-1]
        return df["Low"].iloc[-3] > df["High"].iloc[-1]
    else:
        # Bearish FVG: high[-3] < low[-1]
        return df["High"].iloc[-3] < df["Low"].iloc[-1]


def get_690m_bias():
    """Determine 11.5h bias (BULLISH / BEARISH / RANGING)."""
    df = get_ohlcv_690m(SYMBOL, 10)
    if df is None or len(df) < 3:
        return "RANGING"

    last = df.iloc[-2]   # last closed bar
    prev = df.iloc[-3]

    # Bullish: green bar + liquidity sweep low
    if last["Close"] > last["Open"] and last["Low"] < prev["Low"]:
        return "BULLISH"
    # Bearish: red bar + liquidity sweep high
    elif last["Close"] < last["Open"] and last["High"] > prev["High"]:
        return "BEARISH"

    return "RANGING"


def detect_mss():
    """Market Structure Shift on 1h (break of swing high/low)."""
    df = get_ohlcv(SYMBOL, "1h", 60)
    if df is None or len(df) < 20:
        return False

    price = df["Close"].iloc[-1]
    swing_high = df["High"].iloc[-10:-2].max()
    swing_low = df["Low"].iloc[-10:-2].min()

    return price > swing_high or price < swing_low


def is_weekend():
    """Return True if Saturday (5) or Sunday (6)."""
    now = datetime.now(pytz.timezone(TIMEZONE))
    return now.weekday() >= 5


def is_kill_zone_crypto():
    """
    Crypto-specific Kill Zones (Romania time):

    - Asia:    03:00–06:00
    - London:  09:00–12:00
    - New York core: 15:00–19:00
    """
    now = datetime.now(pytz.timezone(TIMEZONE))
    h = now.hour

    asia = 3 <= h < 6
    london = 9 <= h < 12
    newyork = 15 <= h < 19

    return asia or london or newyork

# ==================== VOLATILITY (ATR) ====================

def compute_atr(df, period=14):
    """
    Simple ATR on given DataFrame.
    df must have High/Low/Close columns.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def passes_volatility_filter(df_5m, price):
    """
    Check if current ATR is within healthy bounds for crypto:
    - not too low (dead market)
    - not too high (crazy spikes, news)
    """
    atr_series = compute_atr(df_5m, ATR_PERIOD)
    atr = atr_series.iloc[-1]

    if pd.isna(atr):
        print("⚠️ ATR insuficient (NaN), sar filtrul de volatilitate.")
        return False

    atr_min = price * ATR_MIN_MULT
    atr_max = price * ATR_MAX_MULT

    print(f"📏 ATR(5m, {ATR_PERIOD}) = {atr:.4f} | Range acceptat: [{atr_min:.4f}, {atr_max:.4f}]")

    if atr < atr_min:
        print("🔻 Volatilitate prea mică (market mort) – sar trade-ul.")
        return False
    if atr > atr_max:
        print("🚨 Volatilitate prea mare (spikes / news) – sar trade-ul.")
        return False

    return True

# ==================== RISK & POSITIONS ====================

def calculate_position_size(balance, risk_pct, entry, sl):
    """
    Position size (qty) in base asset (e.g. BTC for BTC/USDT:USDT) based on risk.
    """
    price_diff = abs(entry - sl)
    if price_diff == 0:
        return 0.0

    risk_amount = balance * (risk_pct / 100.0)
    qty = (risk_amount * LEVERAGE) / (price_diff * entry)
    return float(round(qty, 6))


def get_open_positions_count():
    """Number of active positions (non-zero) on this symbol."""
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

# ==================== ORDER EXECUTION ====================

def execute_trade(direction, entry, sl, tp, risk_pct=None, tag="SWING"):
    """
    Calculate size and send MARKET order on Gate.io Swap.
    risk_pct:
        - None -> folosește RISK_PERCENT global
        - altă valoare -> folosește risc personalizat (ex: scalper)
    tag:
        - "SWING" / "SCALPER" doar pentru log
    """
    try:
        balance = exchange.fetch_balance()
        futures_balance = balance.get(MARGIN_ASSET, {}) or balance.get(f"{MARGIN_ASSET}:{MARGIN_ASSET}", {})
        free_balance = float(futures_balance.get("free", 0))

        if free_balance <= 0:
            print(f"⚠️ [{tag}] Balanță {MARGIN_ASSET} insuficientă.")
            return False

        if risk_pct is None:
            risk_pct = RISK_PERCENT

        qty = calculate_position_size(free_balance, risk_pct, entry, sl)
        if qty <= 0:
            print(f"⚠️ [{tag}] Cantitate invalidă (qty <= 0).")
            return False

        side = "buy" if direction.upper() == "BUY" else "sell"

        print(
            f"📐 [{tag}] Position size: balance={free_balance:.2f} {MARGIN_ASSET}, "
            f"risk={risk_pct:.2f}%, qty={qty}, entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}"
        )
        print(f"📥 [{tag}] Trimit ordin MARKET {side.upper()} pe {SYMBOL} ...")

        order = exchange.create_order(
            symbol=SYMBOL,
            type="market",
            side=side,
            amount=qty
        )
        print(f"✅ [{tag}] Ordin MARKET executat: {order.get('id', 'fără id')}")

        print(f"⚠️ [{tag}] NOTE: SL/TP NU au fost trimise ca ordine pe Gate.io din cod.")
        print("          Setează manual sau adaugă logică suplimentară pentru ordine condiționale.")

        return True

    except Exception as e:
        print(f"❌ [{tag}] Eroare trade: {e}")
        return False

# ==================== SCALPER MODULE ====================

def try_scalper_entry(bias):
    """
    Încearcă un setup de scalper:
    - folosește același bias (BULLISH/BEARISH)
    - timeframe mai mic (SCALPER_TIMEFRAME, ex: 1m)
    - FVG + ATR-based SL
    """
    if not SCALPER_ENABLED:
        return False

    print("⚡ SCALPER: caut setup pe", SCALPER_TIMEFRAME)

    df = get_ohlcv(SYMBOL, SCALPER_TIMEFRAME, 200)
    if df is None or len(df) < SCALPER_ATR_PERIOD + 10:
        print("⚡ SCALPER: date insuficiente pe", SCALPER_TIMEFRAME)
        return False

    price = float(df["Close"].iloc[-1])

    # ATR pentru scalper
    atr_series = compute_atr(df, SCALPER_ATR_PERIOD)
    atr = atr_series.iloc[-1]
    if pd.isna(atr):
        print("⚡ SCALPER: ATR invalid, sar.")
        return False

    # BUY scalper
    if bias == "BULLISH" and detect_fvg(df, bullish=True):
        print("⚡🟢 SCALPER FVG BULLISH pe", SCALPER_TIMEFRAME)
        recent_lows = df["Low"].iloc[-8:]
        swing_low = float(recent_lows.min())
        sl = swing_low - 0.35 * float(atr)       # SL strâns
        tp = float(price + (price - sl) * SCALPER_R_MULT)

        return execute_trade("BUY", price, sl, tp,
                             risk_pct=SCALPER_RISK_PERCENT,
                             tag="SCALPER")

    # SELL scalper
    if bias == "BEARISH" and detect_fvg(df, bullish=False):
        print("⚡🔴 SCALPER FVG BEARISH pe", SCALPER_TIMEFRAME)
        recent_highs = df["High"].iloc[-8:]
        swing_high = float(recent_highs.max())
        sl = swing_high + 0.35 * float(atr)
        tp = float(price - (sl - price) * SCALPER_R_MULT)

        return execute_trade("SELL", price, sl, tp,
                             risk_pct=SCALPER_RISK_PERCENT,
                             tag="SCALPER")

    print("⚡ SCALPER: niciun FVG valid pe", SCALPER_TIMEFRAME)
    return False

# ==================== MAIN TRADING LOOP (ICT 690m CRYPTO) ====================

def ict_690m_bot():
    """Main loop tuned for crypto behavior + scalper."""
    print("=" * 70)
    print("🚀 690m ICT SMART MONEY BOT - Gate.io Futures (CRYPTO + SCALPER)")
    print("=" * 70)

    while True:
        try:
            now = datetime.now(pytz.timezone(TIMEZONE))
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] 🔍 Scan piață pentru {SYMBOL}...")

            # Weekend filter
            if ENABLE_WEEKEND_FILTER and is_weekend():
                print("🛑 Weekend (Sat/Sun) – nu tranzacționez. Revin peste 30 min.")
                time.sleep(1800)
                continue

            # Positions check
            open_pos = get_open_positions_count()
            print(f"📌 Poziții deschise: {open_pos}")
            if open_pos >= MAX_POSITIONS:
                print("⏸️ Există deja poziție activă, aștept 10 minute...")
                time.sleep(600)
                continue

            # 690m Bias
            bias = get_690m_bias()
            print(f"📊 Bias 690m: {bias}")
            if bias == "RANGING":
                print("🔁 Piață în range pe HTF, nu forțez intrări. Revin peste 10 minute.")
                time.sleep(600)
                continue

            # Crypto Kill Zone
            if not is_kill_zone_crypto():
                print("⏰ Nu este Crypto Kill Zone (Asia/London/NY). Revin peste 15 minute.")
                time.sleep(900)
                continue

            # MSS on 1h
            if not detect_mss():
                print("⏳ Aștept MSS pe 1h pentru confirmare. Revin peste 5 minute.")
                time.sleep(300)
                continue

            print("✅ MSS confirmat pe 1h!")

            # Entry on 5m (SWING)
            df_5m = get_ohlcv(SYMBOL, "5m", 150)
            if df_5m is None or len(df_5m) < max(ATR_PERIOD + 5, 40):
                print("⚠️ Date insuficiente pe 5m. Revin peste 2 minute.")
                time.sleep(120)
                continue

            price = float(df_5m["Close"].iloc[-1])

            # Volatility filter (ATR based)
            if ENABLE_VOLATILITY_FILTER:
                if not passes_volatility_filter(df_5m, price):
                    # înainte să sar, încerc scalper
                    if SCALPER_ENABLED:
                        print("⚡ Volatilitate nepotrivită pt swing – încerc SCALPER.")
                        try_scalper_entry(bias)
                    time.sleep(300)
                    continue

            # ATR-based SL (more realistic for crypto)
            atr_series = compute_atr(df_5m, ATR_PERIOD)
            atr = atr_series.iloc[-1]
            if pd.isna(atr):
                print("⚠️ ATR invalid pe 5m, sar runda.")
                time.sleep(300)
                continue

            swing_traded = False

            # BUY swing
            if bias == "BULLISH" and detect_fvg(df_5m, bullish=True):
                print("🟢 SWING FVG BULLISH detectat pe 5m!")
                recent_lows = df_5m["Low"].iloc[-10:]
                swing_low = float(recent_lows.min())
                sl = swing_low - 0.5 * float(atr)
                tp = float(price + (price - sl) * 3.0)  # 3R

                swing_traded = execute_trade("BUY", price, sl, tp,
                                             risk_pct=RISK_PERCENT,
                                             tag="SWING")

            # SELL swing
            elif bias == "BEARISH" and detect_fvg(df_5m, bullish=False):
                print("🔴 SWING FVG BEARISH detectat pe 5m!")
                recent_highs = df_5m["High"].iloc[-10:]
                swing_high = float(recent_highs.max())
                sl = swing_high + 0.5 * float(atr)
                tp = float(price - (sl - price) * 3.0)  # 3R

                swing_traded = execute_trade("SELL", price, sl, tp,
                                             risk_pct=RISK_PERCENT,
                                             tag="SWING")

            else:
                print("⚖️ Niciun setup SWING valid FVG pe 5m în direcția bias-ului.")

            # Dacă nu am făcut trade swing, încerc scalper (dacă e activ)
            if not swing_traded and SCALPER_ENABLED:
                print("⚡ Încerc modul SCALPER...")
                try_scalper_entry(bias)

            # Pauze după ciclul complet
            print("⏳ Aștept 5 minute înainte de următorul scan complet.")
            time.sleep(300)

        except KeyboardInterrupt:
            print("\n⛔ Bot oprit manual (KeyboardInterrupt).")
            break
        except Exception as e:
            print(f"❌ Eroare în loop: {e}")
            print("⏳ Reîncerc în 2 minute...")
            time.sleep(120)

# ==================== MAIN ====================

if __name__ == "__main__":
    try:
        print("🔌 Conectare la Gate.io Futures (swap)...")
        markets = exchange.load_markets()
        if SYMBOL not in markets:
            raise ValueError(
                f"Simbolul {SYMBOL} nu există în markets Gate.io. "
                f"Verifică exact denumirea din fereastra de futures."
            )

        print(f"✅ Conectat la Gate.io ({'TESTNET' if USE_TESTNET else 'LIVE'})")
        print(f"📈 Symbol: {SYMBOL}")

        # Try to set leverage (if supported)
        try:
            exchange.set_leverage(LEVERAGE, SYMBOL)
            print(f"⚙️ Levier setat la {LEVERAGE}x pentru {SYMBOL} (dacă e permis).")
        except Exception as e:
            print(f"⚠️ Nu am putut seta levierul prin API: {e}")
            print("   👉 Asigură-te că levierul este setat manual în interfața Gate.io.")

        ict_690m_bot()

    except Exception as e:
        print(f"❌ Eroare inițializare: {e}")
        print("\n💡 Verifică API Key, Secret, activarea Futures și corectitudinea simbolului.")
