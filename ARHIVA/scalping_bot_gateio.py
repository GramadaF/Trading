import ccxt
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import os
import csv

# ---------------- CONFIG ---------------- #

CONFIG = {
    "exchange": "gateio",              # Gate.io
    "apiKey": "42bb2de8ebdc7c3db84995e203e6752f",
    "secret": "709f16ca0e3da2c8f946f80d336180621ac774aa9aad374a848ddb822975de1a",
    "password": None,
    # Perpetual futures USDT pe Gate.io folosesc formatul: BTC/USDT:USDT
    "symbol": "SOL/USDT:USDT",
    "scalp_tf": "1m",
    "htf_trend_tf": "1h",
    "htf_levels_tf": "15m",
    "scalp_candles": 500,
    "htf_candles": 300,
    "risk_per_trade": 0.5,           # 0.5% din equity
    "leverage": 5,
    "min_atr_percent": 0.05,           # minim 0.05% ATR pentru volatilitate decentă
    "rr": 2.0,                         # Risk:Reward = 1:2
    "max_open_trades": 1,
    "paper_trading": False,             # True = NU trimite ordine reale
    "use_sandbox": False,              # dacă vrei sandbox Gate.io prin ccxt
    "poll_interval_sec": 30,
    "logfile": "scalping_bot_gateio.log",

    # Trailing stop bazat pe ATR
    "use_trailing_atr": True,
    "trailing_atr_mult": 1.5,          # SL urmărește la 1.5 * ATR

    # Jurnal tranzacții
    "journal_csv": "trades_journal.csv"
}

# -------------- LOGGING ----------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["logfile"]),
        logging.StreamHandler()
    ]
)

# ------------- HELPER FUNCS ------------- #

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def local_extrema(df: pd.DataFrame, lookback: int = 5):
    highs = df["high"]
    lows = df["low"]
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        window_high = highs.iloc[i - lookback : i + lookback + 1]
        window_low = lows.iloc[i - lookback : i + lookback + 1]

        if highs.iloc[i] == window_high.max():
            swing_highs.append((df.index[i], float(highs.iloc[i])))

        if lows.iloc[i] == window_low.min():
            swing_lows.append((df.index[i], float(lows.iloc[i])))

    return swing_highs, swing_lows


# ------------- MAIN BOT CLASS ----------- #

class ScalpingBotGateIO:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.exchange = self._init_exchange()
        self.symbol = cfg["symbol"]
        self.open_positions = []  # poziții active (paper + live tracking)

        self._set_leverage_if_possible()
        self._init_journal()

    def _init_exchange(self):
        ex_class = getattr(ccxt, self.cfg["exchange"])
        exchange = ex_class({
            "apiKey": self.cfg["apiKey"],
            "secret": self.cfg["secret"],
            "password": self.cfg["password"],
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap"  # perpetual futures USDT
            }
        })
        if self.cfg.get("use_sandbox", False):
            exchange.set_sandbox_mode(True)
            logging.info("Rulez în sandbox mode Gate.io.")
        return exchange

    def _set_leverage_if_possible(self):
        try:
            if hasattr(self.exchange, "set_leverage"):
                self.exchange.set_leverage(
                    self.cfg["leverage"],
                    self.symbol,
                    params={"reduceOnly": False}
                )
                logging.info(f"Setat leverage {self.cfg['leverage']}x pentru {self.symbol}.")
            else:
                logging.warning("Exchange-ul nu suporta set_leverage prin ccxt.")
        except Exception as e:
            logging.warning(f"Nu am reusit sa setez leverage: {e}")

    def _init_journal(self):
        """
        Creeaza fisierul CSV cu headere daca nu exista.
        """
        path = self.cfg["journal_csv"]
        if not os.path.exists(path):
            with open(path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "opened_at", "closed_at", "direction", "entry", "exit_price",
                    "initial_sl", "final_sl", "tp", "qty",
                    "pnl_usdt", "pnl_rr", "reason"
                ])
            logging.info(f"Creat jurnal de tranzactii: {path}")

    def fetch_ohlcv(self, timeframe: str, limit: int = 300) -> pd.DataFrame:
        data = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    # ---------- PREȚ CURENT ---------- #

    def get_current_price(self) -> float:
        ticker = self.exchange.fetch_ticker(self.symbol)
        return ticker["last"]

    # ---------- STRATEGY BUILDING BLOCKS ---------- #

    def get_htf_bias(self) -> str:
        df = self.fetch_ohlcv(self.cfg["htf_trend_tf"], self.cfg["htf_candles"])
        df["ema50"] = ema(df["close"], 50)
        df["ema200"] = ema(df["close"], 200)

        last = df.iloc[-1]
        if pd.isna(last["ema50"]) or pd.isna(last["ema200"]):
            return "none"

        if last["close"] > last["ema200"] and last["ema50"] > last["ema200"]:
            return "long"
        elif last["close"] < last["ema200"] and last["ema50"] < last["ema200"]:
            return "short"
        else:
            return "none"

    def get_key_levels(self):
        df = self.fetch_ohlcv(self.cfg["htf_levels_tf"], self.cfg["htf_candles"])
        swing_highs, swing_lows = local_extrema(df, lookback=3)

        swing_highs_sorted = sorted(swing_highs, key=lambda x: x[0], reverse=True)[:10]
        swing_lows_sorted = sorted(swing_lows, key=lambda x: x[0], reverse=True)[:10]

        resistances = [price for _, price in swing_highs_sorted]
        supports = [price for _, price in swing_lows_sorted]
        return resistances, supports

    def liquidity_grab_signal(self, bias: str):
        if bias not in ("long", "short"):
            return None

        df = self.fetch_ohlcv(self.cfg["scalp_tf"], self.cfg["scalp_candles"])
        df["atr"] = atr(df, 14)

        last = df.iloc[-1]
        price = last["close"]
        atr_val = last["atr"]
        if pd.isna(atr_val):
            return None

        atr_pct = atr_val / price * 100
        if atr_pct < self.cfg["min_atr_percent"]:
            logging.info(f"ATR prea mic ({atr_pct:.4f}%), piata lenta - nu intru.")
            return None

        resistances, supports = self.get_key_levels()

        c_open = last["open"]
        c_high = last["high"]
        c_low = last["low"]
        c_close = last["close"]
        timestamp = df.index[-1]

        # LONG – liquidity grab sub suport
        if bias == "long":
            for sup in supports:
                if c_low < sup and c_close > sup and c_close > c_open:
                    sl = c_low - atr_val * 0.2
                    risk = c_close - sl
                    tp = c_close + risk * self.cfg["rr"]
                    return {
                        "direction": "long",
                        "entry": float(c_close),
                        "sl": float(sl),
                        "tp": float(tp),
                        "level": float(sup),
                        "time": timestamp
                    }

        # SHORT – liquidity grab peste rezistență
        if bias == "short":
            for res in resistances:
                if c_high > res and c_close < res and c_close < c_open:
                    sl = c_high + atr_val * 0.2
                    risk = sl - c_close
                    tp = c_close - risk * self.cfg["rr"]
                    return {
                        "direction": "short",
                        "entry": float(c_close),
                        "sl": float(sl),
                        "tp": float(tp),
                        "level": float(res),
                        "time": timestamp
                    }

        return None

    # ---------- ORDER MANAGEMENT ---------- #

    def get_balance(self) -> float:
        balance = self.exchange.fetch_balance()

        usdt = None
        if "USDT" in balance:
            entry = balance["USDT"]
            if isinstance(entry, dict):
                usdt = entry.get("free") or entry.get("total")
            else:
                usdt = entry

        if not usdt and "total" in balance and isinstance(balance["total"], dict):
            usdt = balance["total"].get("USDT", 0)

        return usdt or 0.0

    def calc_position_size(self, entry: float, sl: float) -> float:
        equity = self.get_balance()
        risk_amount = equity * self.cfg["risk_per_trade"]
        stop_distance = abs(entry - sl)
        if stop_distance == 0:
            return 0.0
        qty = (risk_amount / stop_distance) * self.cfg["leverage"]
        return float(qty)

    def open_order(self, signal: dict):
        direction = signal["direction"]
        entry = signal["entry"]
        sl = signal["sl"]
        tp = signal["tp"]

        qty = self.calc_position_size(entry, sl)
        if qty <= 0:
            logging.warning("Qty calculata = 0. Nu deschid.")
            return

        side = "buy" if direction == "long" else "sell"

        logging.info(
            f"📥 Signal {direction.upper()} | entry={entry:.4f}, sl={sl:.4f}, tp={tp:.4f}, qty={qty:.4f}"
        )

        position = {
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "initial_sl": sl,
            "tp": tp,
            "qty": qty,
            "status": "open",
            "opened_at": datetime.utcnow(),
            "exchange_order_id": None,
            "highest_price": entry,
            "lowest_price": entry
        }

        if self.cfg["paper_trading"]:
            self.open_positions.append(position)
            logging.info("Paper trade salvat (nu am trimis ordin real Gate.io).")
            return

        # LIVE ORDERS – ATENȚIE, TESTEAZĂ PE SUME FOARTE MICI!
        params = {}
        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=side,
                amount=qty,
                params=params
            )
            position["exchange_order_id"] = order.get("id")
            self.open_positions.append(position)
            logging.info(f"Ordin deschis pe Gate.io: {order}")
            logging.info("SL/TP + trailing vor fi gestionate intern de bot (market close).")
        except Exception as e:
            logging.error(f"Eroare la trimitere ordin Gate.io: {e}")

    # ---------- JURNAL & ÎNCHIDERE POZIȚII ---------- #

    def _log_trade(self, pos: dict, exit_price: float, reason: str):
        pnl_usdt = 0.0
        if pos["direction"] == "long":
            pnl_usdt = (exit_price - pos["entry"]) * pos["qty"]
        else:
            pnl_usdt = (pos["entry"] - exit_price) * pos["qty"]

        # Calcul R
        risk_per_unit = abs(pos["entry"] - pos["initial_sl"])
        risk_amount = risk_per_unit * pos["qty"] if risk_per_unit > 0 else None
        pnl_rr = pnl_usdt / risk_amount if risk_amount and risk_amount != 0 else None

        with open(self.cfg["journal_csv"], mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                pos.get("opened_at"),
                pos.get("closed_at"),
                pos["direction"],
                pos["entry"],
                exit_price,
                pos["initial_sl"],
                pos["sl"],
                pos["tp"],
                pos["qty"],
                pnl_usdt,
                pnl_rr,
                reason
            ])

        logging.info(
            f"📝 Trade log: dir={pos['direction']} entry={pos['entry']:.4f} exit={exit_price:.4f} "
            f"pnl={pnl_usdt:.4f} USDT, R={pnl_rr:.2f if pnl_rr is not None else 'NA'}"
        )

    def close_position(self, pos: dict, reason: str, exit_price: float | None = None):
        if pos["status"] != "open":
            return

        if exit_price is None:
            try:
                exit_price = self.get_current_price()
            except Exception as e:
                logging.error(f"Nu am reusit sa iau pretul de iesire: {e}")
                return

        direction = pos["direction"]
        qty = pos["qty"]

        logging.info(
            f"🔚 Inchid pozitie {direction.upper()} (qty={qty:.4f}) motiv: {reason.upper()} "
            f"la pretul {exit_price:.4f}"
        )

        if self.cfg["paper_trading"]:
            pos["status"] = "closed"
            pos["closed_at"] = datetime.utcnow()
            pos["close_reason"] = reason
            self._log_trade(pos, exit_price, reason)
            return

        # LIVE: trimite market order în sens invers
        side = "sell" if direction == "long" else "buy"

        params = {}
        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=side,
                amount=qty,
                params=params
            )
            logging.info(f"Ordin de închidere trimis: {order}")
            pos["status"] = "closed"
            pos["closed_at"] = datetime.utcnow()
            pos["close_reason"] = reason

            # Poți lua prețul din order["average"] dacă vrei mai precis
            self._log_trade(pos, exit_price, reason)
        except Exception as e:
            logging.error(f"Eroare la închiderea pozitiei: {e}")

    # ---------- TRAILING STOP ATR + MONITORIZARE ---------- #

    def _apply_trailing_atr(self, pos: dict, price: float, atr_val: float):
        """
        Ajusteaza SL in functie de ATR, doar in directia profitului.
        """
        if not self.cfg.get("use_trailing_atr", False):
            return
        if np.isnan(atr_val):
            return

        mult = self.cfg["trailing_atr_mult"]
        direction = pos["direction"]

        if direction == "long":
            # actualizează highest_price
            pos["highest_price"] = max(pos.get("highest_price", pos["entry"]), price)
            proposed_sl = pos["highest_price"] - atr_val * mult

            # SL mai sus (mai aproape), dar sub prețul curent
            if proposed_sl > pos["sl"] and proposed_sl < price:
                logging.info(
                    f"🔄 Trailing ATR LONG: mut SL {pos['sl']:.4f} -> {proposed_sl:.4f}"
                )
                pos["sl"] = proposed_sl

        elif direction == "short":
            pos["lowest_price"] = min(pos.get("lowest_price", pos["entry"]), price)
            proposed_sl = pos["lowest_price"] + atr_val * mult

            # SL mai jos (mai aproape), dar deasupra prețului curent
            if proposed_sl < pos["sl"] and proposed_sl > price:
                logging.info(
                    f"🔄 Trailing ATR SHORT: mut SL {pos['sl']:.4f} -> {proposed_sl:.4f}"
                )
                pos["sl"] = proposed_sl

    def monitor_positions(self):
        """
        Verifica pretul curent, aplica trailing ATR si inchide pozitiile la SL/TP.
        """
        if not self.open_positions:
            return

        try:
            price = self.get_current_price()
        except Exception as e:
            logging.error(f"Nu am reusit sa iau pretul curent: {e}")
            return

        # Calculăm ATR o singură dată pentru TF de scalping
        atr_val = None
        if self.cfg.get("use_trailing_atr", False):
            try:
                df = self.fetch_ohlcv(self.cfg["scalp_tf"], self.cfg["scalp_candles"])
                df["atr"] = atr(df, 14)
                atr_val = float(df["atr"].iloc[-1])
            except Exception as e:
                logging.error(f"Eroare la calcul ATR pentru trailing: {e}")
                atr_val = None

        for pos in self.open_positions:
            if pos["status"] != "open":
                continue

            # aplică trailing ATR dacă avem ATR
            if atr_val is not None:
                self._apply_trailing_atr(pos, price, atr_val)

            direction = pos["direction"]
            sl = pos["sl"]
            tp = pos["tp"]

            # LONG: TP când price >= tp, SL când price <= sl
            if direction == "long":
                if price >= tp:
                    logging.info(f"✅ TP atins pentru LONG: price={price:.4f} ≥ tp={tp:.4f}")
                    self.close_position(pos, reason="tp", exit_price=price)
                elif price <= sl:
                    logging.info(f"❌ SL lovit pentru LONG: price={price:.4f} ≤ sl={sl:.4f}")
                    self.close_position(pos, reason="sl", exit_price=price)

            # SHORT: TP când price <= tp, SL când price >= sl
            elif direction == "short":
                if price <= tp:
                    logging.info(f"✅ TP atins pentru SHORT: price={price:.4f} ≤ tp={tp:.4f}")
                    self.close_position(pos, reason="tp", exit_price=price)
                elif price >= sl:
                    logging.info(f"❌ SL lovit pentru SHORT: price={price:.4f} ≥ sl={sl:.4f}")
                    self.close_position(pos, reason="sl", exit_price=price)

        # opțional poți curăța pozițiile închise:
        # self.open_positions = [p for p in self.open_positions if p["status"] == "open"]

    # ---------- MAIN LOOP ---------- #

    def run(self):
        logging.info(f"🚀 Pornit ScalpingBotGateIO pe {self.symbol}.")
        while True:
            try:
                # Monitorizează întâi pozițiile existente (trailing + SL/TP)
                self.monitor_positions()

                open_count = sum(1 for p in self.open_positions if p["status"] == "open")
                if open_count >= self.cfg["max_open_trades"]:
                    logging.info("Ai deja numarul maxim de pozitii deschise. Astept...")
                    time.sleep(self.cfg["poll_interval_sec"])
                    continue

                bias = self.get_htf_bias()
                logging.info(f"HTF bias: {bias}")
                if bias == "none":
                    logging.info("Fara trend clar HTF - nu iau trade-uri.")
                    time.sleep(self.cfg["poll_interval_sec"])
                    continue

                signal = self.liquidity_grab_signal(bias)
                if signal:
                    logging.info(f"✅ Signal gasit: {signal}")
                    self.open_order(signal)
                else:
                    logging.info("Niciun signal valid pe LTF.")

            except Exception as e:
                logging.error(f"Eroare in loop: {e}")

            time.sleep(self.cfg["poll_interval_sec"])


if __name__ == "__main__":
    bot = ScalpingBotGateIO(CONFIG)
    bot.run()
