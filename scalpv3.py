# ======================================================================
#                           ScalpV4
# ======================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ccxt
import time
import json
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import csv
import sys

from dotenv import load_dotenv
from telegram_notifier import TelegramNotifier

# ======================================================================
# LOAD CONFIG FROM .env
# ======================================================================

load_dotenv()

def env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except:
        return default

def env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except:
        return default

CONFIG = {
    "exchange": os.getenv("EXCHANGE", "gateio"),

    "apiKey": os.getenv("GATEIO_API_KEY"),
    "secret": os.getenv("GATEIO_API_SECRET"),
    "password": None,

    "symbol": os.getenv("SYMBOL", "SOL/USDT:USDT"),

    "scalp_tf": os.getenv("SCALP_TF", "1m"),
    "htf_trend_tf": os.getenv("HTF_TREND_TF", "1h"),
    "htf_levels_tf": os.getenv("HTF_LEVELS_TF", "15m"),
    "scalp_candles": env_int("SCALP_CANDLES", 300),
    "htf_candles": env_int("HTF_CANDLES", 300),

    "risk_per_trade": env_float("RISK_PER_TRADE", 0.008),
    "leverage": env_int("LEVERAGE", 3),
    "max_open_trades": env_int("MAX_OPEN_TRADES", 1),

    "min_atr_percent": env_float("MIN_ATR_PERCENT", 0.09),
    "rr": env_float("RR", 1.3),
    "min_sl_percent": env_float("MIN_SL_PERCENT", 0.003),

    "use_trailing_atr": env_bool("USE_TRAILING_ATR", True),
    "trailing_atr_mult": env_float("TRAILING_ATR_MULT", 1.4),

    "allowed_hours": [
        (9, 12),
        (13, 16)
    ],

    "news_blackout": [
        (2, 15, 16),
        (4, 15, 16),
    ],

    "max_consecutive_losses": 2,
    "loss_cooldown_minutes": 60,
    "low_atr_pause_minutes": 15,

    "max_spread_percent": env_float("MAX_SPREAD_PERCENT", 0.05),

    "paper_trading": env_bool("PAPER_TRADING", False),
    "use_sandbox": env_bool("USE_SANDBOX", False),

    "poll_interval_sec": env_int("POLL_INTERVAL_SEC", 10),
    "logfile": "scalping_bot_gateio.log",
    "journal_csv": "trades_journal.csv",

    "max_down_time_sec": env_int("MAX_DOWN_TIME_SEC", 300),
}

if not CONFIG["apiKey"] or not CONFIG["secret"]:
    raise RuntimeError("Lipsesc GATEIO_API_KEY / GATEIO_API_SECRET în .env")

# ======================================================================
# LOGGING
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["logfile"]),
        logging.StreamHandler()
    ]
)

# ======================================================================
# HELPERS
# ======================================================================

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
        w_high = highs.iloc[i - lookback : i + lookback + 1]
        w_low  = lows.iloc[i - lookback : i + lookback + 1]

        if highs.iloc[i] == w_high.max():
            swing_highs.append((df.index[i], float(highs.iloc[i])))

        if lows.iloc[i] == w_low.min():
            swing_lows.append((df.index[i], float(lows.iloc[i])))

    return swing_highs, swing_lows

# ======================================================================
# BOT
# ======================================================================

class ScalpingBotGateIO:
    def __init__(self, cfg: dict, notifier: TelegramNotifier):
        self.cfg = cfg
        self.notifier = notifier
        self.symbol = cfg["symbol"]

        self.exchange = self._init_exchange()
        self.open_positions = []

        self.consecutive_losses = 0
        self.loss_cooldown_until = None
        self.last_low_atr_time = None

        self.last_ok = time.time()
        self.max_down_time = cfg["max_down_time_sec"]

        # Telegram updates offset
        self.last_update_id = None

        # used for 16:01 journal sending
        self.last_journal_sent_date = None

        self._set_leverage()
        self._init_journal()

    # ----------------------------------------------------------
    # Internal notifier helpers
    # ----------------------------------------------------------

    def _notify(self, msg):
        if self.notifier:
            self.notifier.send(msg)

    def _notify_exc(self, prefix, exc):
        if self.notifier:
            self.notifier.send_exception(prefix, exc)

    # ----------------------------------------------------------

    def _mark_ok(self):
        self.last_ok = time.time()

    def _check_health(self):
        if time.time() - self.last_ok > self.max_down_time:
            msg = "⛔ Exchange down >5min. Restart systemd."
            logging.error(msg)
            self._notify(msg)
            raise SystemExit(1)

    # ----------------------------------------------------------
    # Init exchange
    # ----------------------------------------------------------

    def _init_exchange(self):
        ex_class = getattr(ccxt, self.cfg["exchange"])
        ex = ex_class({
            "apiKey": self.cfg["apiKey"],
            "secret": self.cfg["secret"],
            "password": self.cfg["password"],
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        if self.cfg["use_sandbox"]:
            ex.set_sandbox_mode(True)
        try:
            ex.load_markets()
            self._mark_ok()
        except Exception as e:
            logging.error("load_markets error:", e)
        return ex

    def _set_leverage(self):
        try:
            self.exchange.set_leverage(self.cfg["leverage"], self.symbol)
            self._mark_ok()
        except:
            pass

    def _init_journal(self):
        path = self.cfg["journal_csv"]
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "opened_at","closed_at","direction","entry","exit_price",
                    "initial_sl","final_sl","tp","qty","pnl_usdt","pnl_rr","reason"
                ])
            logging.info("Journal created.")

    # ======================================================================
    # FILE UPLOAD (CSV+JSON) at 16:01 RO
    # ======================================================================

    def _send_journals_if_1601(self):
        now_ro = datetime.utcnow() + timedelta(hours=2)
        if now_ro.hour == 16 and now_ro.minute == 1:
            today_str = now_ro.strftime("%Y-%m-%d")

            if self.last_journal_sent_date == today_str:
                return  # already sent today

            csv_path = self.cfg["journal_csv"]
            json_path = "trades_journal.json"

            # convert CSV -> JSON
            data = []
            if os.path.exists(csv_path):
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        data.append(row)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            self._notify("📑 Trimit jurnalele de tranzacții...")

            # send CSV
            self.notifier.send_file(csv_path, caption="Jurnal CSV")

            # send JSON
            self.notifier.send_file(json_path, caption="Jurnal JSON")

            self._notify("✅ Jurnalele au fost trimise.")

            self.last_journal_sent_date = today_str

    # ======================================================================
    # OHLCV, PRICE
    # ======================================================================

    def fetch_ohlcv(self, tf, limit):
        data = self.exchange.fetch_ohlcv(self.symbol, timeframe=tf, limit=limit)
        self._mark_ok()
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def get_price(self):
        t = self.exchange.fetch_ticker(self.symbol)
        self._mark_ok()
        return float(t["last"])

    # ======================================================================
    # STRATEGY
    # ======================================================================

    def get_htf_bias(self):
        df = self.fetch_ohlcv(self.cfg["htf_trend_tf"], self.cfg["htf_candles"])
        df["ema50"] = ema(df["close"], 50)
        df["ema200"] = ema(df["close"], 200)
        last = df.iloc[-1]

        if last["close"] > last["ema200"] and last["ema50"] > last["ema200"]:
            return "long"
        if last["close"] < last["ema200"] and last["ema50"] < last["ema200"]:
            return "short"
        return "none"

    def get_key_levels(self):
        df = self.fetch_ohlcv(self.cfg["htf_levels_tf"], self.cfg["htf_candles"])
        swing_h, swing_l = local_extrema(df, 3)
        swing_h = sorted(swing_h, key=lambda x: x[0], reverse=True)[:10]
        swing_l = sorted(swing_l, key=lambda x: x[0], reverse=True)[:10]
        return [p for _, p in swing_h], [p for _, p in swing_l]

    def liquidity_signal(self, bias):
        if bias not in ("long","short"):
            return None

        df = self.fetch_ohlcv(self.cfg["scalp_tf"], self.cfg["scalp_candles"])
        df["atr"] = atr(df, 14)
        last = df.iloc[-1]

        price = float(last["close"])
        atr_val = float(last["atr"])
        if np.isnan(atr_val):
            return None

        atr_pct = atr_val / price * 100
        if atr_pct < self.cfg["min_atr_percent"]:
            logging.info(f"ATR mic {atr_pct:.4f}%")
            self.last_low_atr_time = datetime.utcnow()
            return None
        else:
            self.last_low_atr_time = None

        resist, supp = self.get_key_levels()

        o = float(last["open"])
        h = float(last["high"])
        l = float(last["low"])
        c = float(last["close"])

        min_sl = price * self.cfg["min_sl_percent"]
        timestamp = df.index[-1]

        if bias == "long":
            for s in supp:
                if l < s < c and c > o:
                    raw_sl = l - atr_val * 0.2
                    sl = min(raw_sl, c - min_sl)
                    risk = c - sl
                    if risk > 0:
                        tp = c + risk * self.cfg["rr"]
                        return {"direction":"long","entry":c,"sl":sl,"tp":tp,"level":s,"time":timestamp}

        if bias == "short":
            for r in resist:
                if h > r > c and c < o:
                    raw_sl = h + atr_val*0.2
                    sl = max(raw_sl, c + min_sl)
                    risk = sl - c
                    if risk > 0:
                        tp = c - risk*self.cfg["rr"]
                        return {"direction":"short","entry":c,"sl":sl,"tp":tp,"level":r,"time":timestamp}

        return None

    # ======================================================================
    # BALANCE
    # ======================================================================

    def get_balance(self):
        b = self.exchange.fetch_balance()
        self._mark_ok()
        usdt = b.get("USDT", {})
        return float(usdt.get("free", 0.0))

    def normalize_qty(self, qty):
        try:
            m = self.exchange.market(self.symbol)
            self._mark_ok()
        except:
            return 0

        min_amount = m.get("limits", {}).get("amount", {}).get("min", 0.0)
        try:
            q = float(self.exchange.amount_to_precision(self.symbol, qty))
        except:
            q = qty
        if q < min_amount:
            return 0
        return q

    def calc_qty(self, entry, sl):
        balance = self.get_balance()
        if balance <= 0:
            return 0
        risk_amount = balance * self.cfg["risk_per_trade"]
        stop_dist = abs(entry - sl)
        if stop_dist == 0:
            return 0

        qty_risk = (risk_amount / stop_dist) * self.cfg["leverage"]
        qty_max = (balance * self.cfg["leverage"]) / entry
        qty = min(qty_risk, qty_max)
        if qty <= 0:
            return 0

        if self.cfg["paper_trading"]:
            return qty
        return self.normalize_qty(qty)

    # ======================================================================
    # ORDERS
    # ======================================================================

    def open_order(self, sig):
        direction = sig["direction"]
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]

        qty = self.calc_qty(entry, sl)
        if qty <= 0:
            logging.warning("Qty invalida.")
            return

        side = "buy" if direction=="long" else "sell"
        pos = {
            "direction":direction,
            "entry":entry,
            "sl":sl,
            "initial_sl":sl,
            "tp":tp,
            "qty":qty,
            "status":"open",
            "opened_at":datetime.utcnow(),
            "exchange_order_id":None,
            "highest_price":entry,
            "lowest_price":entry
        }

        self._notify(
            f"📈 Pozitie {direction.upper()} deschisa\n"
            f"Entry: {entry}\nSL: {sl}\nTP: {tp}\nQty: {qty}\n"
        )

        if self.cfg["paper_trading"]:
            self.open_positions.append(pos)
            return

        try:
            o = self.exchange.create_order(self.symbol, "market", side, qty)
            pos["exchange_order_id"] = o.get("id")
            self.open_positions.append(pos)
            self._mark_ok()
        except Exception as e:
            logging.error("Order error:", e)
            self._notify_exc("Order error:", e)

    def close_position(self, pos, reason, exit_price=None):
        if pos["status"] != "open":
            return

        if exit_price is None:
            try:
                exit_price = self.get_price()
            except:
                return

        direction = pos["direction"]
        side = "sell" if direction=="long" else "buy"
        qty = pos["qty"]

        if self.cfg["paper_trading"]:
            pos["status"]="closed"
            pos["closed_at"]=datetime.utcnow()
            self._log_trade(pos, exit_price, reason)
            return

        try:
            o = self.exchange.create_order(self.symbol, "market", side, qty, params={"reduceOnly":True})
            pos["status"]="closed"
            pos["closed_at"]=datetime.utcnow()
            self._log_trade(pos, exit_price, reason)
            self._mark_ok()
        except Exception as e:
            logging.error("Close error:", e)
            self._notify_exc("Close error:", e)

    # ======================================================================
    # TRADE LOG
    # ======================================================================

    def _log_trade(self, pos, exit_price, reason):
        if pos["direction"]=="long":
            pnl = (exit_price - pos["entry"]) * pos["qty"]
        else:
            pnl = (pos["entry"] - exit_price) * pos["qty"]

        risk_unit = abs(pos["entry"] - pos["initial_sl"])
        risk = risk_unit * pos["qty"] if risk_unit>0 else None
        rr = pnl/risk if risk else None

        with open(self.cfg["journal_csv"], "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                pos["opened_at"], pos["closed_at"], pos["direction"], pos["entry"],
                exit_price, pos["initial_sl"], pos["sl"], pos["tp"], pos["qty"], pnl, rr, reason
            ])

        self._notify(
            f"📉 Pozitie {pos['direction'].upper()} inchisa\n"
            f"Exit: {exit_price}\nPnL: {pnl:.4f}\nR: {rr}"
        )

    # ======================================================================
    # TELEGRAM COMMANDS (/status, /pnl)
    # ======================================================================

    def _handle_status(self):
        try:
            price = self.get_price()
        except:
            price = None

        now_ro = datetime.utcnow() + timedelta(hours=2)
        open_count = sum(1 for p in self.open_positions if p["status"]=="open")

        msg = (
            "📊 STATUS BOT\n"
            f"Symbol: {self.symbol}\n"
            f"Time RO: {now_ro}\n"
            f"Open positions: {open_count}\n"
            f"Paper: {self.cfg['paper_trading']}\n"
        )
        if price: msg += f"Price: {price}\n"
        self._notify(msg)

    def _handle_pnl(self):
        csv_path = self.cfg["journal_csv"]
        if not os.path.exists(csv_path):
            self._notify("❗Nu exista jurnal CSV.")
            return

        total_pnl = 0
        count = 0

        with open(csv_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                try:
                    pnl = float(row["pnl_usdt"])
                    total_pnl += pnl
                    count += 1
                except:
                    pass

        self._notify(
            f"💰 PNL TOTAL\nTrades: {count}\nPnL: {total_pnl:.4f} USDT"
        )

    def _poll_telegram(self):
        if not self.notifier or not self.notifier.config.enabled:
            return

        updates = self.notifier.get_updates(offset=None if self.last_update_id is None else self.last_update_id+1)
        for upd in updates:
            uid = upd.get("update_id")
            if uid:
                self.last_update_id = uid

            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue

            text = msg.get("text", "").strip()
            if not text:
                continue

            if text.startswith("/status"):
                self._handle_status()
            elif text.startswith("/pnl"):
                self._handle_pnl()

    # ======================================================================
    # MONITOR TRAILING
    # ======================================================================

    def _apply_trailing(self, pos, price, atr_val):
        if not self.cfg["use_trailing_atr"]:
            return
        if np.isnan(atr_val):
            return

        mult = self.cfg["trailing_atr_mult"]
        if pos["direction"]=="long":
            pos["highest_price"] = max(pos["highest_price"], price)
            new_sl = pos["highest_price"] - atr_val*mult
            if new_sl > pos["sl"] and new_sl < price:
                pos["sl"] = new_sl
        else:
            pos["lowest_price"] = min(pos["lowest_price"], price)
            new_sl = pos["lowest_price"] + atr_val*mult
            if new_sl < pos["sl"] and new_sl > price:
                pos["sl"] = new_sl

    def _monitor_positions(self):
        if not self.open_positions:
            return

        try:
            price = self.get_price()
        except:
            return

        try:
            df = self.fetch_ohlcv(self.cfg["scalp_tf"], self.cfg["scalp_candles"])
            df["atr"] = atr(df, 14)
            atr_val = float(df["atr"].iloc[-1])
        except:
            atr_val = None

        for pos in self.open_positions:
            if pos["status"]!="open":
                continue

            if atr_val:
                self._apply_trailing(pos, price, atr_val)

            if pos["direction"]=="long":
                if price >= pos["tp"]:
                    self.close_position(pos, "tp", price)
                elif price <= pos["sl"]:
                    self.close_position(pos, "sl", price)
            else:
                if price <= pos["tp"]:
                    self.close_position(pos, "tp", price)
                elif price >= pos["sl"]:
                    self.close_position(pos, "sl", price)

    # ======================================================================
    # SPREAD FILTER
    # ======================================================================

    def _spread_ok(self):
        max_spread = self.cfg["max_spread_percent"]
        try:
            ob = self.exchange.fetch_order_book(self.symbol, limit=5)
            self._mark_ok()
            b = ob.get("bids", [])
            a = ob.get("asks", [])
            if not b or not a:
                return False
            bid = float(b[0][0])
            ask = float(a[0][0])
            mid = (bid+ask)/2
            spread = (ask-bid)/mid*100
            return spread <= max_spread
        except:
            return False

    # ======================================================================
    # MAIN LOOP
    # ======================================================================

    def run(self):
        self._notify(f"🚀 Bot pornit pe {self.symbol}, lev {self.cfg['leverage']}x")

        while True:
            try:
                # send journals if time = 16:01
                self._send_journals_if_1601()

                # poll telegram commands
                self._poll_telegram()

                # monitor open positions
                self._monitor_positions()

                # time filters
                now_utc = datetime.utcnow()
                now_ro = now_utc + timedelta(hours=2)
                wd = now_ro.weekday()
                hr = now_ro.hour

                # weekend
                if wd in (5,6):
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_health()
                    continue

                # allowed hours
                if not any(start<=hr<end for start,end in self.cfg["allowed_hours"]):
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_health()
                    continue

                # news blackout
                if any(wd==d and start<=hr<end for d,start,end in self.cfg["news_blackout"]):
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_health()
                    continue

                # cooldown after consecutive losses
                if self.loss_cooldown_until and now_utc < self.loss_cooldown_until:
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_health()
                    continue

                # low atr cooldown
                if self.last_low_atr_time:
                    if now_utc - self.last_low_atr_time < timedelta(minutes=self.cfg["low_atr_pause_minutes"]):
                        time.sleep(self.cfg["poll_interval_sec"])
                        self._check_health()
                        continue

                # spread filter
                if not self._spread_ok():
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_health()
                    continue

                # open positions count
                if sum(1 for p in self.open_positions if p["status"]=="open") >= self.cfg["max_open_trades"]:
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_health()
                    continue

                # bias
                bias = self.get_htf_bias()
                logging.info(f"Bias: {bias}")
                if bias=="none":
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_health()
                    continue

                # signal
                sig = self.liquidity_signal(bias)
                if sig:
                    self.open_order(sig)
                else:
                    logging.info("Fara signal.")

                time.sleep(self.cfg["poll_interval_sec"])
                self._check_health()

            except SystemExit:
                raise
            except Exception as e:
                logging.error("Loop error:", e)
                self._notify_exc("Loop error:", e)
                time.sleep(self.cfg["poll_interval_sec"])
                self._check_health()


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    notifier = TelegramNotifier()
    bot = ScalpingBotGateIO(CONFIG, notifier)
    bot.run()
