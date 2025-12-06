# ======================================================================
#                           ScalpV4
# ======================================================================

import ccxt
import time
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import csv
import sys

from dotenv import load_dotenv
from telegram_notifier import TelegramNotifier

load_dotenv()

# ---------------- CONFIG ---------------- #

CONFIG = {
    "exchange": "gateio",

    # cheile vin din .env
    "apiKey": os.getenv("GATEIO_API_KEY"),
    "secret": os.getenv("GATEIO_API_SECRET"),
    "password": None,

    # symbol din .env, cu fallback
    "symbol": os.getenv("SYMBOL", "SOL/USDT:USDT"),

    # Timeframe-uri
    "scalp_tf": "1m",
    "htf_trend_tf": "1h",
    "htf_levels_tf": "15m",
    "scalp_candles": 300,
    "htf_candles": 300,

    # 📌 Risk management (cont ~80–100 USDT)
    "risk_per_trade": 0.008,          # 0.8% din cont / trade
    "leverage": 3,                    # levier stabil 3x
    "max_open_trades": 1,             # doar 1 pozitie simultan

    # ⚡ Volatilitate minima (ATR) si R:R
    "min_atr_percent": 0.09,          # 0.09% ATR minim (evita piata moarta)
    "rr": 1.3,                        # R:R stabil pentru scalping 1m

    # 🛡 SL minim ca procent din pret
    "min_sl_percent": 0.0030,         # 0.30% din pret (SL mai sanatos)

    # 🎯 Trailing stop bazat pe ATR
    "use_trailing_atr": True,
    "trailing_atr_mult": 1.4,         # trailing un pic mai lejer decat 1.2

    # ⏰ Intervalele orare de trading (ora Romaniei)
    "allowed_hours": [
        (9, 12),   # 09:00 - 11:59 (London)
        (13, 16),  # 13:00 - 15:59 (pre-New York)
    ],

    # 🚫 Intervalele de blackout pentru "stiri" (ora Romaniei)
    # (weekday, start_hour, end_hour) - 0=Luni ... 6=Duminica
    "news_blackout": [
        (2, 15, 16),  # Miercuri 15:00-15:59 (CPI/FOMC de obicei)
        (4, 15, 16),  # Vineri 15:00-15:59 (NFP)
    ],

    # 😬 Pauza dupa pierderi consecutive
    "max_consecutive_losses": 2,
    "loss_cooldown_minutes": 60,      # 1 ora pauza dupa 2 SL-uri la rand

    # 😴 Pauza cand ATR este mic (piata moarta)
    "low_atr_pause_minutes": 15,      # asteapta 15 min dupa ATR sub prag

    # 📉 Filtru de spread
    "max_spread_percent": 0.05,       # 0.05% spread maxim acceptat

    # 🔧 Rulare / logging
    "paper_trading": False,           # True pentru test, False pentru live
    "use_sandbox": False,
    "poll_interval_sec": 10,
    "logfile": "scalping_bot_gateio.log",
    "journal_csv": "trades_journal.csv",

    # 🔔 Telegram din .env
    "telegram_token": os.getenv("TELEGRAM_TOKEN"),
    "telegram_chat_id": int(os.getenv("TELEGRAM_CHAT_ID", "0")) or None,

    # ❤️ Health-check: daca nu avem conexiune ok la exchange X secunde => exit(1) => systemd restart
    "max_down_time_sec": 60 * 5,      # 5 minute
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

if not CONFIG["apiKey"] or not CONFIG["secret"]:
    raise RuntimeError("Lipsesc GATEIO_API_KEY / GATEIO_API_SECRET din .env")

# ------------- TELEGRAM NOTIFIER -------- #

notifier = TelegramNotifier(
    token=CONFIG.get("telegram_token", ""),
    chat_id=CONFIG.get("telegram_chat_id", ""),
    enabled=bool(CONFIG.get("telegram_token") and CONFIG.get("telegram_chat_id")),
    timeout=5,
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
    def __init__(self, cfg: dict, notifier_obj: TelegramNotifier = None):
        self.cfg = cfg
        self.notifier = notifier_obj
        self.symbol = cfg["symbol"]

        self.exchange = self._init_exchange()
        self.open_positions = []  # pozitii active (paper + live tracking)

        # tracking pentru filtre avansate
        self.consecutive_losses = 0
        self.loss_cooldown_until = None
        self.last_low_atr_time = None

        # health-check conexiune
        self.last_ok = time.time()
        self.max_down_time = cfg.get("max_down_time_sec", 60 * 5)

        # Telegram commands
        self.last_telegram_update_id = None

        # pentru fallback la balance
        self.last_balance = 0.0

        self._set_leverage_if_possible()
        self._init_journal()

    # --------- NOTIFIER HELPERS --------- #

    def _notify(self, msg: str):
        if self.notifier:
            self.notifier.send(msg)

    def _notify_exception(self, prefix: str, exc: BaseException):
        if self.notifier:
            self.notifier.send_exception(prefix, exc)

    # --------- HEALTH-CHECK HELPERS ----- #

    def _mark_exchange_ok(self):
        self.last_ok = time.time()

    def _check_exchange_health(self):
        delta = time.time() - self.last_ok
        if delta > self.max_down_time:
            mins = int(self.max_down_time / 60)
            msg = (
                f"⛔ Nu am mai avut conexiune corecta la exchange de peste {mins} minute. "
                f"Ies si las systemd sa ma reporneasca."
            )
            logging.error(msg)
            self._notify(msg)
            raise SystemExit(1)

    # --------- TELEGRAM COMENZI --------- #

    def _handle_status_command(self):
        try:
            price = None
            try:
                price = self.get_current_price()
            except Exception as e:
                logging.error(f"Nu am putut lua pretul pentru /status: {e}")

            now_utc = datetime.utcnow()
            now_ro = now_utc + timedelta(hours=2)
            open_count = sum(1 for p in self.open_positions if p["status"] == "open")

            msg = (
                "📊 Status bot:\n"
                f"Symbol: {self.symbol}\n"
                f"Time RO: {now_ro.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Open positions: {open_count}\n"
                f"Paper trading: {self.cfg['paper_trading']}\n"
            )
            if price is not None:
                msg += f"Last price: {price}\n"

            self._notify(msg)
        except Exception as e:
            logging.error(f"Eroare la /status: {e}")
            self._notify(f"❗ Eroare la /status: {e}")

    def _poll_telegram_commands(self):
        if not self.notifier or not self.cfg.get("telegram_chat_id"):
            return

        offset = self.last_telegram_update_id + 1 if self.last_telegram_update_id is not None else None
        updates = self.notifier.get_updates(offset=offset, limit=20)

        for upd in updates:
            uid = upd.get("update_id")
            if uid is not None:
                self.last_telegram_update_id = uid

            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue

            chat = msg.get("chat", {})
            if chat.get("id") != self.cfg["telegram_chat_id"]:
                continue

            text = (msg.get("text") or "").strip()
            if not text:
                continue

            if text.startswith("/status"):
                self._handle_status_command()

    # ------------- INIT/SETUP ----------- #

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
            logging.info("Rulez in sandbox mode Gate.io.")
        try:
            exchange.load_markets()
            self._mark_exchange_ok()
        except Exception as e:
            logging.error(f"Eroare la load_markets: {e}")
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
                self._mark_exchange_ok()
            else:
                logging.warning("Exchange-ul nu suporta set_leverage prin ccxt.")
        except Exception as e:
            logging.warning(f"Nu am reusit sa setez leverage: {e}")

    def _init_journal(self):
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

    # ---------- OHLCV / PRETURI ---------- #

    def fetch_ohlcv(self, timeframe: str, limit: int = 300) -> pd.DataFrame:
        data = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)
        self._mark_exchange_ok()
        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def get_current_price(self) -> float:
        ticker = self.exchange.fetch_ticker(self.symbol)
        self._mark_exchange_ok()
        return float(ticker["last"])

    # ---------- STRATEGIE: BIAS + LEVELS + LIQ GRAB ---------- #

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
        price = float(last["close"])
        atr_val = float(last["atr"])
        if pd.isna(atr_val):
            return None

        atr_pct = atr_val / price * 100.0
        if atr_pct < self.cfg["min_atr_percent"]:
            logging.info(f"ATR prea mic ({atr_pct:.4f}%), piata lenta - nu intru.")
            self.last_low_atr_time = datetime.utcnow()
            return None
        else:
            self.last_low_atr_time = None

        resistances, supports = self.get_key_levels()

        c_open = float(last["open"])
        c_high = float(last["high"])
        c_low = float(last["low"])
        c_close = float(last["close"])
        timestamp = df.index[-1]

        min_sl_percent = self.cfg.get("min_sl_percent", 0.0015)
        min_sl_distance = price * min_sl_percent

        if bias == "long":
            for sup in supports:
                if c_low < sup and c_close > sup and c_close > c_open:
                    raw_sl = c_low - atr_val * 0.2
                    min_allowed_sl = c_close - min_sl_distance
                    sl = min(raw_sl, min_allowed_sl)

                    risk = c_close - sl
                    if risk <= 0:
                        logging.warning(
                            f"Risk calculat <= 0 la LONG (entry={c_close}, sl={sl}). Sar peste semnal."
                        )
                        continue

                    tp = c_close + risk * self.cfg["rr"]

                    return {
                        "direction": "long",
                        "entry": c_close,
                        "sl": float(sl),
                        "tp": float(tp),
                        "level": float(sup),
                        "time": timestamp
                    }

        if bias == "short":
            for res in resistances:
                if c_high > res and c_close < res and c_close < c_open:
                    raw_sl = c_high + atr_val * 0.2
                    min_allowed_sl = c_close + min_sl_distance
                    sl = max(raw_sl, min_allowed_sl)

                    risk = sl - c_close
                    if risk <= 0:
                        logging.warning(
                            f"Risk calculat <= 0 la SHORT (entry={c_close}, sl={sl}). Sar peste semnal."
                        )
                        continue

                    tp = c_close - risk * self.cfg["rr"]

                    return {
                        "direction": "short",
                        "entry": c_close,
                        "sl": float(sl),
                        "tp": float(tp),
                        "level": float(res),
                        "time": timestamp
                    }

        return None

    # ---------- BALANTA / POSITION SIZE ---------- #

    def get_balance(self) -> float:
        """
        Balanță pentru futures USDT (swap): equity în USDT.
        Folosește fetch_balance(type='swap') și fallback la ultima valoare.
        """
        try:
            balance = self.exchange.fetch_balance({'type': 'swap'})
            self._mark_exchange_ok()

            usdt_info = balance.get("USDT", {})
            if not isinstance(usdt_info, dict):
                logging.error("Format neasteptat pentru balance['USDT'].")
                return self.last_balance

            total = float(usdt_info.get("total", 0.0) or 0.0)
            if total < 0:
                logging.warning("Equity negativ detectat. Setez la 0.")
                total = 0.0

            self.last_balance = total
            return total

        except Exception as e:
            logging.error(f"[ERROR] get_balance() exception: {e}")
            if self.last_balance is not None:
                logging.warning(f"Folosesc fallback last_balance={self.last_balance}")
                return self.last_balance
            return 0.0

    def normalize_qty_to_market(self, qty: float) -> float:
        try:
            market = self.exchange.market(self.symbol)
            self._mark_exchange_ok()
        except Exception as e:
            logging.error(f"Nu am putut lua market info pentru {self.symbol}: {e}")
            return 0.0

        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {}) if limits else {}
        min_amount = amount_limits.get("min", 0.0) or 0.0

        try:
            qty_precise = float(self.exchange.amount_to_precision(self.symbol, qty))
        except Exception:
            qty_precise = qty

        if qty_precise < min_amount:
            logging.warning(
                f"Qty calculata ({qty_precise}) este sub min_amount ({min_amount}) pentru {self.symbol}. "
                f"Nu trimit ordinul."
            )
            return 0.0

        return qty_precise

    def calc_position_size(self, entry: float, sl: float) -> float:
        """
        Position sizing SAFE pentru futures:
        - folosește equity din get_balance()
        - corectează formula de risc (fără levier dublu)
        - respectă MAX_QTY din .env
        - respectă min_amount + precision prin normalize_qty_to_market
        """
        try:
            equity = self.get_balance()
            if equity <= 0:
                logging.warning("Equity <= 0. Nu pot deschide pozitii.")
                return 0.0

            risk_percent = float(self.cfg.get("risk_per_trade", 0.005))
            leverage = float(self.cfg.get("leverage", 3))

            risk_amount = equity * risk_percent
            if risk_amount <= 0:
                logging.warning("Risk amount <= 0. Nu pot calcula marimea pozitiei.")
                return 0.0

            stop_distance = abs(entry - sl)
            if stop_distance < 0.0005:
                logging.warning(f"SL prea mic (distanta={stop_distance:.6f}). Ajustez la 0.0005.")
                stop_distance = 0.0005

            # risk_amount ≈ stop_distance * qty * leverage  => qty = risk_amount / (stop_distance * leverage)
            qty = risk_amount / (stop_distance * leverage)

            # optional: limitare dupa marja maxima teoretica
            max_notional = equity * leverage
            qty_max_margin = max_notional / entry
            if qty > qty_max_margin:
                logging.info(
                    f"Qty {qty:.4f} depaseste qty_max_margin {qty_max_margin:.4f}. Ajustez la qty_max_margin."
                )
                qty = qty_max_margin

            if qty <= 0:
                logging.warning("Qty calculata <= 0 dupa limitarile de risc/margin.")
                return 0.0

            # aplicam MAX_QTY din .env
            max_qty_env = os.getenv("MAX_QTY")
            if max_qty_env is not None:
                try:
                    max_qty = float(max_qty_env)
                    if qty > max_qty:
                        logging.info(
                            f"Qty {qty:.4f} > MAX_QTY {max_qty:.4f}. Ajustez la MAX_QTY."
                        )
                        qty = max_qty
                except ValueError:
                    logging.warning(f"Valoare invalida pentru MAX_QTY: {max_qty_env}")

            if self.cfg.get("paper_trading", True):
                return float(qty)

            qty = self.normalize_qty_to_market(qty)
            return qty

        except Exception as e:
            logging.error(f"[ERROR] calc_position_size() exception: {e}")
            return 0.0

    # ---------- DESCHIDERE POZITII ---------- #

    def open_order(self, signal: dict):
        direction = signal["direction"]
        entry = signal["entry"]
        sl = signal["sl"]
        tp = signal["tp"]

        qty = self.calc_position_size(entry, sl)
        if qty <= 0:
            logging.warning("Qty calculata este prea mica sau nu respecta limitarile. Nu deschid pozitie.")
            return

        # safety: verificam MAX_QTY inca o data
        max_qty_env = os.getenv("MAX_QTY")
        if max_qty_env is not None:
            try:
                max_qty = float(max_qty_env)
                if qty > max_qty:
                    logging.info(
                        f"Qty {qty:.4f} > MAX_QTY {max_qty:.4f} (hard check). Ajustez la MAX_QTY."
                    )
                    qty = max_qty
            except ValueError:
                logging.warning(f"Valoare invalida pentru MAX_QTY: {max_qty_env}")

        side = "buy" if direction == "long" else "sell"

        logging.info(
            f"Signal {direction.upper()} | entry={entry:.4f}, sl={sl:.4f}, tp={tp:.4f}, qty={qty:.4f}"
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

        self._notify(
            f"📈 Pozitie {direction.upper()} deschisa\n"
            f"Symbol: {self.symbol}\n"
            f"Entry: {entry:.4f}\n"
            f"SL: {sl:.4f}\n"
            f"TP: {tp:.4f}\n"
            f"Qty: {qty:.4f}\n"
            f"Paper: {self.cfg['paper_trading']}"
        )

        if self.cfg["paper_trading"]:
            self.open_positions.append(position)
            logging.info("Paper trade salvat (nu am trimis ordin real Gate.io).")
            return

        params = {}
        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=side,
                amount=qty,
                params=params
            )
            self._mark_exchange_ok()
            position["exchange_order_id"] = order.get("id")
            self.open_positions.append(position)
            logging.info(f"Ordin deschis pe Gate.io: {order}")
            logging.info("SL/TP + trailing sunt gestionate intern de bot (inchidere market).")
        except Exception as e:
            logging.error(f"Eroare la trimitere ordin Gate.io: {e}")
            self._notify(f"❌ Eroare la trimitere ordin Gate.io: {e}")

    # ---------- JURNAL / INCHIDERE POZITII ---------- #

    def _log_trade(self, pos: dict, exit_price: float, reason: str):
        if pos["direction"] == "long":
            pnl_usdt = (exit_price - pos["entry"]) * pos["qty"]
        else:
            pnl_usdt = (pos["entry"] - exit_price) * pos["qty"]

        risk_per_unit = abs(pos["entry"] - pos["initial_sl"])
        risk_amount = risk_per_unit * pos["qty"] if risk_per_unit > 0 else None
        pnl_rr = pnl_usdt / risk_amount if risk_amount and risk_amount != 0 else None

        if pnl_usdt < 0:
            self.consecutive_losses += 1
        elif pnl_usdt > 0:
            self.consecutive_losses = 0

        max_losses = self.cfg.get("max_consecutive_losses", 2)
        if self.consecutive_losses >= max_losses:
            minutes = self.cfg.get("loss_cooldown_minutes", 60)
            self.loss_cooldown_until = datetime.utcnow() + timedelta(minutes=minutes)
            logging.info(
                f"S-au inregistrat {self.consecutive_losses} pierderi consecutive. "
                f"Pauza de intrari pentru urmatoarele {minutes} minute."
            )
            self.consecutive_losses = 0

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
            f"Trade log: dir={pos['direction']} entry={pos['entry']:.4f} exit={exit_price:.4f} "
            f"pnl={pnl_usdt:.4f} USDT, R={pnl_rr if pnl_rr is not None else 'NA'}"
        )

        self._notify(
            f"📉 Pozitie {pos['direction'].upper()} inchisa ({reason})\n"
            f"Entry: {pos['entry']:.4f}\n"
            f"Exit: {exit_price:.4f}\n"
            f"Qty: {pos['qty']:.4f}\n"
            f"PnL: {pnl_usdt:.4f} USDT\n"
            f"R: {pnl_rr if pnl_rr is not None else 'NA'}"
        )

    def close_position(self, pos: dict, reason: str, exit_price: float = None):
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
            f"Inchid pozitie {direction.upper()} (qty={qty:.4f}) motiv: {reason.upper()} "
            f"la pretul {exit_price:.4f}"
        )

        if self.cfg["paper_trading"]:
            pos["status"] = "closed"
            pos["closed_at"] = datetime.utcnow()
            pos["close_reason"] = reason
            self._log_trade(pos, exit_price, reason)
            return

        side = "sell" if direction == "long" else "buy"
        params = {
            "reduceOnly": True
        }

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=side,
                amount=qty,
                params=params
            )
            self._mark_exchange_ok()
            logging.info(f"Ordin de inchidere trimis: {order}")
            pos["status"] = "closed"
            pos["closed_at"] = datetime.utcnow()
            pos["close_reason"] = reason
            self._log_trade(pos, exit_price, reason)
        except Exception as e:
            logging.error(f"Eroare la inchiderea pozitiei: {e}")
            self._notify(f"❌ Eroare la inchiderea pozitiei: {e}")

    # ---------- TRAILING STOP ATR + MONITORIZARE ---------- #

    def _apply_trailing_atr(self, pos: dict, price: float, atr_val: float):
        if not self.cfg.get("use_trailing_atr", False):
            return
        if np.isnan(atr_val):
            return

        mult = self.cfg["trailing_atr_mult"]
        direction = pos["direction"]

        if direction == "long":
            pos["highest_price"] = max(pos.get("highest_price", pos["entry"]), price)
            proposed_sl = pos["highest_price"] - atr_val * mult

            if proposed_sl > pos["sl"] and proposed_sl < price:
                logging.info(
                    f"Trailing ATR LONG: mut SL {pos['sl']:.4f} -> {proposed_sl:.4f}"
                )
                pos["sl"] = proposed_sl

        elif direction == "short":
            pos["lowest_price"] = min(pos.get("lowest_price", pos["entry"]), price)
            proposed_sl = pos["lowest_price"] + atr_val * mult

            if proposed_sl < pos["sl"] and proposed_sl > price:
                logging.info(
                    f"Trailing ATR SHORT: mut SL {pos['sl']:.4f} -> {proposed_sl:.4f}"
                )
                pos["sl"] = proposed_sl

    def monitor_positions(self):
        if not self.open_positions:
            return

        try:
            price = self.get_current_price()
        except Exception as e:
            logging.error(f"Nu am reusit sa iau pretul curent: {e}")
            return

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

            if atr_val is not None:
                self._apply_trailing_atr(pos, price, atr_val)

            direction = pos["direction"]
            sl = pos["sl"]
            tp = pos["tp"]

            if direction == "long":
                if price >= tp:
                    logging.info(f"TP atins pentru LONG: price={price:.4f} >= tp={tp:.4f}")
                    self.close_position(pos, reason="tp", exit_price=price)
                elif price <= sl:
                    logging.info(f"SL lovit pentru LONG: price={price:.4f} <= sl={sl:.4f}")
                    self.close_position(pos, reason="sl", exit_price=price)

            elif direction == "short":
                if price <= tp:
                    logging.info(f"TP atins pentru SHORT: price={price:.4f} <= tp={tp:.4f}")
                    self.close_position(pos, reason="tp", exit_price=price)
                elif price >= sl:
                    logging.info(f"SL lovit pentru SHORT: price={price:.4f} >= sl={sl:.4f}")
                    self.close_position(pos, reason="sl", exit_price=price)

    # ---------- SPREAD / LICHIDITATE ---------- #

    def spread_ok(self) -> bool:
        max_spread_pct = self.cfg.get("max_spread_percent", 0.1)

        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=5)
            self._mark_exchange_ok()
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            if not bids or not asks:
                logging.warning("Order book gol sau incomplet - nu tranzactionez.")
                return False

            bid = float(bids[0][0])
            ask = float(asks[0][0])
            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid * 100.0

            if spread_pct > max_spread_pct:
                logging.info(
                    f"Spread prea mare: {spread_pct:.4f}% (> {max_spread_pct}%). Sar peste intrari."
                )
                return False

            return True

        except Exception as e:
            logging.error(f"Eroare la verificarea spread-ului: {e}")
            return False

    # ---------- MAIN LOOP ---------- #

    def run(self):
        logging.info(f"Pornit ScalpingBotGateIO pe {self.symbol}.")
        self._notify(f"🚀 ScalpingBotGateIO a pornit pe {self.symbol} (leverage {self.cfg['leverage']}x).")

        while True:
            try:
                self.monitor_positions()

                now_utc = datetime.utcnow()
                now_ro = now_utc + timedelta(hours=2)
                ro_weekday = now_ro.weekday()
                ro_hour = now_ro.hour

                if ro_weekday in (5, 6):
                    logging.info("Weekend detectat - nu deschid noi pozitii.")
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_exchange_health()
                    self._poll_telegram_commands()
                    continue

                allowed = False
                for start, end in self.cfg.get("allowed_hours", []):
                    if start <= ro_hour < end:
                        allowed = True
                        break

                if not allowed:
                    logging.info(
                        f"Este ora {ro_hour}:00 RO - in afara intervalului orar - nu caut intrari."
                    )
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_exchange_health()
                    self._poll_telegram_commands()
                    continue

                blackout_list = self.cfg.get("news_blackout", [])
                in_blackout = False
                for wday, start, end in blackout_list:
                    if ro_weekday == wday and start <= ro_hour < end:
                        in_blackout = True
                        break

                if in_blackout:
                    logging.info(
                        "Suntem in interval de stiri (blackout) - nu caut intrari."
                    )
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_exchange_health()
                    self._poll_telegram_commands()
                    continue

                if self.loss_cooldown_until is not None and now_utc < self.loss_cooldown_until:
                    minutes_left = (self.loss_cooldown_until - now_utc).total_seconds() / 60.0
                    logging.info(
                        f"Inca in cooldown dupa pierderi consecutive "
                        f"(~{minutes_left:.1f} minute ramase) - nu caut intrari."
                    )
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_exchange_health()
                    self._poll_telegram_commands()
                    continue

                low_atr_pause = self.cfg.get("low_atr_pause_minutes", 15)
                if self.last_low_atr_time is not None:
                    dt = now_utc - self.last_low_atr_time
                    if dt < timedelta(minutes=low_atr_pause):
                        logging.info(
                            f"ATR a fost prea mic recent (acum {dt.total_seconds()/60:.1f} min) - "
                            f"astept inainte de noi intrari."
                        )
                        time.sleep(self.cfg["poll_interval_sec"])
                        self._check_exchange_health()
                        self._poll_telegram_commands()
                        continue

                if not self.spread_ok():
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_exchange_health()
                    self._poll_telegram_commands()
                    continue

                open_count = sum(1 for p in self.open_positions if p["status"] == "open")
                if open_count >= self.cfg["max_open_trades"]:
                    logging.info("Numarul maxim de pozitii deschise atins. Astept...")
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_exchange_health()
                    self._poll_telegram_commands()
                    continue

                bias = self.get_htf_bias()
                logging.info(f"HTF bias: {bias}")
                if bias == "none":
                    logging.info("Fara trend clar HTF - nu iau trade-uri.")
                    time.sleep(self.cfg["poll_interval_sec"])
                    self._check_exchange_health()
                    self._poll_telegram_commands()
                    continue

                signal = self.liquidity_grab_signal(bias)
                if signal:
                    logging.info(f"Signal gasit: {signal}")
                    self.open_order(signal)
                else:
                    logging.info("Niciun signal valid pe LTF.")

                time.sleep(self.cfg["poll_interval_sec"])
                self._check_exchange_health()
                self._poll_telegram_commands()

            except SystemExit as e:
                raise
            except Exception as e:
                logging.error(f"Eroare in loop: {e}")
                self._notify(f"⚠️ Eroare in loop: {e}")
                time.sleep(self.cfg["poll_interval_sec"])
                self._check_exchange_health()
                self._poll_telegram_commands()


if __name__ == "__main__":
    bot = ScalpingBotGateIO(CONFIG, notifier_obj=notifier)
    try:
        bot.run()
    except SystemExit as e:
        sys.exit(e.code if isinstance(e.code, int) else 1)
    except Exception as e:
        logging.error("💀 Botul a crapat cu exceptie necontrolata.", exc_info=True)
        if notifier:
            notifier.send_exception("💀 Botul a crapat cu exceptie necontrolata:", e)
        sys.exit(1)
