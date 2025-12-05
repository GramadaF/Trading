import os
import time
import math
import logging
from datetime import datetime

import ccxt
import pandas as pd


class ScalpingBotV3:
    def __init__(self):
        self.load_env()
        self.setup_logging()
        self.setup_exchange()
        self.last_balance = 0.0
        self.current_position = None

    # ---------------- ENV & LOGGING & EXCHANGE ----------------

    def load_env(self):
        # Chei API
        self.api_key = os.getenv("API_KEY", "")
        self.api_secret = os.getenv("API_SECRET", "")

        # Symbol – tu poti pune in .env: SYMBOL=XRP/USDT:USDT sau SOL/USDT:USDT
        self.symbol = os.getenv("SYMBOL", "XRP/USDT:USDT")

        # Parametri principali
        self.leverage = float(os.getenv("LEVERAGE", "3"))
        self.risk_per_trade = float(os.getenv("RISK_PER_TRADE", "0.005"))  # 0.5%
        self.min_atr_percent = float(os.getenv("MIN_ATR_PERCENT", "0.08"))
        self.min_sl_percent = float(os.getenv("MIN_SL_PERCENT", "0.003"))  # 0.30%
        self.rr = float(os.getenv("RR", "1.3"))
        self.poll_interval = int(os.getenv("POLL_INTERVAL_SEC", "60"))

        # Limite cantitate
        max_qty_env = os.getenv("MAX_QTY")
        self.max_qty = float(max_qty_env) if max_qty_env is not None else None
        self.symbol_min_qty = float(os.getenv("MIN_QTY", "10"))  # minim tehnic (XRP: 10)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        self.logger = logging.getLogger("ScalpV3")

    def log(self, msg: str):
        self.logger.info(msg)

    def setup_exchange(self):
        self.exchange = ccxt.gateio({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",  # USDT-M futures (swap)
            },
        })
        self.exchange.load_markets()

        try:
            # setare levier unde e suportat
            self.exchange.set_leverage(int(self.leverage), self.symbol)
            self.log(f"Leverage set to {self.leverage}x on {self.symbol}")
        except Exception as e:
            self.log(f"Could not set leverage on {self.symbol}: {e}")

    # ---------------- BALANCE SAFE ----------------

    def get_balance(self) -> float:
        """
        Returneaza equity pentru Gate.io USDT-M Futures:
        available + unrealised pnl, cu fallback in caz de eroare.
        """
        try:
            acc = self.exchange.private_futures_get_settle_accounts({'settle': 'usdt'})
            if not acc or 'available' not in acc[0]:
                self.log("[ERROR] API returned invalid balance format.")
                return getattr(self, "last_balance", 0.0)

            available = float(acc[0].get("available", 0.0))
            unrealized = float(acc[0].get("unrealised_pnl", 0.0))
            equity = available + unrealized

            if equity < 0:
                self.log("[WARNING] Negative equity detected. Setting to 0.")
                equity = 0.0

            self.last_balance = equity
            return equity

        except Exception as e:
            self.log(f"[ERROR] get_balance() exception: {e}")
            if hasattr(self, "last_balance"):
                self.log(f"[WARNING] Using fallback last_balance: {self.last_balance}")
                return self.last_balance
            return 0.0

    # ---------------- POSITION SIZING SAFE ----------------

    def calculate_position_size(self, entry: float, sl: float) -> float:
        """
        Calculeaza qty in mod sigur pentru cont mic:
        - foloseste equity real
        - limiteaza SL prea mic
        - aplica min_qty
        - aplica MAX_QTY din .env
        """
        try:
            balance = self.get_balance()
            risk_percent = float(self.risk_per_trade)
            leverage = float(self.leverage)

            risk_amount = balance * risk_percent
            if risk_amount <= 0:
                self.log("[ERROR] Risk amount <= 0. Return 0.")
                return 0.0

            stop_distance = abs(entry - sl)
            if stop_distance < 0.0005:
                stop_distance = 0.0005
                self.log(f"[WARNING] SL too small. stop_distance adjusted to {stop_distance}")

            # qty brut
            qty = (risk_amount / stop_distance) / leverage

            # rotunjire pentru XRP (1 decimal)
            qty = float(f"{qty:.1f}")

            # min_qty tehnic (XRP futures ~10)
            min_qty = self.symbol_min_qty
            if qty < min_qty:
                self.log(f"[INFO] Qty {qty} < min_qty {min_qty}. Adjust to min_qty.")
                qty = min_qty

            # limita HARD din .env
            if self.max_qty is not None and qty > self.max_qty:
                self.log(f"[WARNING] Qty {qty} > MAX_QTY {self.max_qty}. Reduce to MAX_QTY.")
                qty = self.max_qty

            if qty <= 0:
                self.log("[ERROR] Final qty <= 0. Return 0.")
                return 0.0

            return qty

        except Exception as e:
            self.log(f"[ERROR] calculate_position_size() exception: {e}")
            return 0.0

    # ---------------- ORDER OPEN SAFE ----------------

    def open_order(self, side: str, entry: float, sl: float, tp: float):
        """
        Deschide ordinul principal, cu qty sigura.
        Aplica inca o data MAX_QTY ca limită hard, apoi trimite create_order().
        """
        try:
            qty = self.calculate_position_size(entry, sl)

            # limită hard încă o dată, pentru siguranță
            if self.max_qty is not None and qty > self.max_qty:
                self.log(f"[WARNING] Qty {qty} > MAX_QTY {self.max_qty} (hard). Adjust to MAX_QTY.")
                qty = self.max_qty

            if qty <= 0:
                self.log("[ERROR] Qty invalid (<=0). Not sending order.")
                return None

            self.log(f"[INFO] Sending {side.upper()} qty={qty} entry={entry} sl={sl} tp={tp}")

            order = self.exchange.create_order(
                self.symbol,
                "market",
                side,
                qty,
            )

            self.log(f"[SUCCESS] Opened order: {order}")

            self.current_position = {
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "order": order,
            }

            return order

        except Exception as e:
            self.log(f"[ERROR] open_order() error: {e}")
            return None

    # ---------------- MARKET DATA & ATR / BIAS ----------------

    def fetch_ohlcv(self, timeframe: str = "1m", limit: int = 200):
        return self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        return atr

    def get_bias_and_atr(self):
        ohlcv = self.fetch_ohlcv("1m", 200)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)

        close = df["close"]
        ema50 = close.ewm(span=50).mean()
        ema200 = close.ewm(span=200).mean()

        if ema50.iloc[-1] > ema200.iloc[-1]:
            bias = "long"
        elif ema50.iloc[-1] < ema200.iloc[-1]:
            bias = "short"
        else:
            bias = "none"

        atr = self.compute_atr(df).iloc[-1]
        atr_percent = (atr / close.iloc[-1]) * 100.0

        return bias, float(atr_percent), df

    # ---------------- SEMNALE (placeholder simplu) ----------------

    def find_signal(self, bias: str, df: pd.DataFrame):
        """
        Placeholder simplu de semnal.
        In momentul de fata NU deschide poziti automat.
        Daca vrei, aici poti implementa logica ta (liquidity grab etc.).
        Return:
            dict cu keys: side, entry, sl, tp
            sau None daca nu exista semnal.
        """
        return None

    # ---------------- MAIN LOOP ----------------

    def run(self):
        self.log(f"ScalpingBotV3 started on {self.symbol} (lev {self.leverage}x).")
        while True:
            try:
                bias, atr_percent, df = self.get_bias_and_atr()

                self.log(f"Bias: {bias}")
                if atr_percent < self.min_atr_percent:
                    self.log(f"ATR mic {atr_percent:.4f}%")
                    self.log("Fara signal.")
                else:
                    self.log(f"ATR {atr_percent:.4f}%")
                    signal = self.find_signal(bias, df)
                    if signal:
                        side = signal["side"]
                        entry = signal["entry"]
                        sl = signal["sl"]
                        tp = signal["tp"]
                        self.open_order(side, entry, sl, tp)
                    else:
                        self.log("Fara signal.")

            except Exception as e:
                self.log(f"[ERROR] run() loop error: {e}")

            time.sleep(self.poll_interval)


if __name__ == "__main__":
    bot = ScalpingBotV3()
    bot.run()