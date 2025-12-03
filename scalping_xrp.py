import ccxt
import time
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import csv

# ---------------- CONFIG ---------------- #

CONFIG = {
    "exchange": "gateio",
    "apiKey": "42bb2de8ebdc7c3db84995e203e6752f",
    "secret": "709f16ca0e3da2c8f946f80d336180621ac774aa9aad374a848ddb822975de1a",
    "password": None,

    # Perechea de futures (swap USDT) Gate.io - XRP acum
    "symbol": "XRP/USDT:USDT",

    # Timeframe-uri
    "scalp_tf": "1m",          # scalping pe 1 minut
    "htf_trend_tf": "1h",      # bias pe 1h
    "htf_levels_tf": "15m",    # levels pe 15m
    "scalp_candles": 300,
    "htf_candles": 300,

    # 📌 Risk management (cont ~40–100 USDT)
    "risk_per_trade": 0.008,   # 0.8% din cont / trade
    "leverage": 3,             # levier stabil 3x
    "max_open_trades": 1,      # o singura pozitie simultan (XRP se misca repede)

    # ⚡ Volatilitate minima (ATR) si R:R
    # XRP este mai "nervos" decat SOL, dar tot vrem sa evitam range-urile moarte
    "min_atr_percent": 0.08,   # putin mai relaxat decat 0.09, ca sa prinda miscari XRP
    "rr": 1.3,                 # RR stabil pentru scalping

    # 🛡 SL minim ca procent din pret
    # XRP este ieftin (~0.5–0.7 USDT), 0.30% ramane OK
    "min_sl_percent": 0.0030,  # 0.30% din pret

    # 🎯 Trailing stop bazat pe ATR
    "use_trailing_atr": True,
    "trailing_atr_mult": 1.4,  # trailing destul de lejer, dar protejeaza profitul

    # ⏰ Intervalele orare de trading (ora Romaniei)
    "allowed_hours": [
        (9, 12),   # 09:00 - 11:59 (London)
        (13, 16),  # 13:00 - 15:59 (pre-New York)
    ],

    # 🚫 Intervalele de blackout pentru "stiri" (ora Romaniei)
    "news_blackout": [
        (2, 15, 16),  # Miercuri 15:00-15:59 (CPI/FOMC de obicei)
        (4, 15, 16),  # Vineri 15:00-15:59 (NFP)
    ],

    # 😬 Pauza dupa pierderi consecutive
    "max_consecutive_losses": 2,
    "loss_cooldown_minutes": 60,  # 1 ora pauza dupa 2 SL-uri la rand

    # 😴 Pauza cand ATR este mic (piata moarta)
    "low_atr_pause_minutes": 15,

    # 📉 Filtru de spread
    "max_spread_percent": 0.05,   # XRP are spread mic, 0.05% e safe

    # 🔧 Rulare / logging
    "paper_trading": True,        # lasa TRUE pana testam bine pe XRP
    "use_sandbox": False,
    "poll_interval_sec": 10,
    "logfile": "scalping_bot_gateio_xrp.log",
    "journal_csv": "trades_journal_xrp.csv",
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
        self.open_positions = []  # pozitii active (paper + live tracking)

        # tracking pentru filtre avansate
        self.consecutive_losses = 0
        self.loss_cooldown_until = None
        self.last_low_atr_time = None

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
            logging.info("Rulez in sandbox mode Gate.io.")
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

    # ---------- OHLCV / PRETURI ---------- #

    def fetch_ohlcv(self, timeframe: str, limit: int = 300) -> pd.DataFrame:
        data = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def get_current_price(self) -> float:
        ticker = self.exchange.fetch_ticker(self.symbol)
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
        """
        Detecteaza liquidity grab pe timeframe-ul de scalping.
        Include:
        - filtru ATR minim
        - distanta minima SL (min_sl_percent)
        - memorare cand ATR este prea mic (last_low_atr_time)
        """
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

        # BIAS LONG: liquidity grab sub suport
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

        # BIAS SHORT: liquidity grab peste rezistenta
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
        Returneaza strict USDT disponibil (free), nu total.
        """
        balance = self.exchange.fetch_balance()

        usdt_free = 0.0
        if "USDT" in balance and isinstance(balance["USDT"], dict):
            usdt_free = balance["USDT"].get("free", 0.0) or 0.0

        return float(usdt_free)

    def normalize_qty_to_market(self, qty: float) -> float:
        """
        Ajusteaza qty la limitarile de pe exchange (min amount + precizie).
        Daca dupa ajustare qty < min_amount => intoarce 0 si nu mai trimitem ordinul.
        """
        try:
            market = self.exchange.market(self.symbol)
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
        Calculeaza marimea pozitiei pe baza riscului % si o limiteaza la marginul disponibil.
        """
        equity_free = self.get_balance()
        if equity_free <= 0:
            logging.warning("Nu exista USDT disponibil (free) in cont. Nu pot deschide pozitii.")
            return 0.0

        stop_distance = abs(entry - sl)
        if stop_distance == 0:
            logging.warning("Stop distance = 0. Nu pot calcula marimea pozitiei.")
            return 0.0

        risk_amount = equity_free * self.cfg["risk_per_trade"]
        qty_risk_based = (risk_amount / stop_distance) * self.cfg["leverage"]

        max_notional = equity_free * self.cfg["leverage"]
        qty_max_margin = max_notional / entry

        raw_qty = min(qty_risk_based, qty_max_margin)

        if raw_qty <= 0:
            logging.warning("Qty calculata este <= 0 dupa limitarile de risc/margin.")
            return 0.0

        # daca suntem in paper trading, putem sari normalizarea pentru a vedea "teoretic" ce ar face
        if self.cfg.get("paper_trading", True):
            return float(raw_qty)

        qty = self.normalize_qty_to_market(raw_qty)
        return qty

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
            position["exchange_order_id"] = order.get("id")
            self.open_positions.append(position)
            logging.info(f"Ordin deschis pe Gate.io: {order}")
            logging.info("SL/TP + trailing sunt gestionate intern de bot (inchidere market).")
        except Exception as e:
            logging.error(f"Eroare la trimitere ordin Gate.io: {e}")

    # ---------- JURNAL / INCHIDERE POZITII ---------- #

    def _log_trade(self, pos: dict, exit_price: float, reason: str):
        if pos["direction"] == "long":
            pnl_usdt = (exit_price - pos["entry"]) * pos["qty"]
        else:
            pnl_usdt = (pos["entry"] - exit_price) * pos["qty"]

        risk_per_unit = abs(pos["entry"] - pos["initial_sl"])
        risk_amount = risk_per_unit * pos["qty"] if risk_per_unit > 0 else None
        pnl_rr = pnl_usdt / risk_amount if risk_amount and risk_amount != 0 else None

        # update pentru pierderi consecutive
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
            logging.info(f"Ordin de inchidere trimis: {order}")
            pos["status"] = "closed"
            pos["closed_at"] = datetime.utcnow()
            pos["close_reason"] = reason

            # putem folosi exit_price calculat de noi sau order["average"]
            self._log_trade(pos, exit_price, reason)
        except Exception as e:
            logging.error(f"Eroare la inchiderea pozitiei: {e}")

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
        """
        Verifica daca spread-ul nu este prea mare.
        Foloseste top of book din order book.
        """
        max_spread_pct = self.cfg.get("max_spread_percent", 0.1)

        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=5)
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
        while True:
            try:
                # 1) Monitorizam mereu pozitiile existente (chiar si in weekend / noaptea)
                self.monitor_positions()

                now_utc = datetime.utcnow()
                now_ro = now_utc + timedelta(hours=2)  # Romania ~ UTC+2
                ro_weekday = now_ro.weekday()          # 0=Luni, 6=Duminica
                ro_hour = now_ro.hour

                # 2) Filtru WEEKEND pentru intrari noi
                if ro_weekday in (5, 6):
                    logging.info("Weekend detectat - nu deschid noi pozitii.")
                    time.sleep(self.cfg["poll_interval_sec"])
                    continue

                # 3) Filtru ORAR (allowed_hours)
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
                    continue

                # 4) Filtru de "stiri" (news_blackout)
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
                    continue

                # 5) Pauza dupa pierderi consecutive
                if self.loss_cooldown_until is not None and now_utc < self.loss_cooldown_until:
                    minutes_left = (self.loss_cooldown_until - now_utc).total_seconds() / 60.0
                    logging.info(
                        f"Inca in cooldown dupa pierderi consecutive "
                        f"(~{minutes_left:.1f} minute ramase) - nu caut intrari."
                    )
                    time.sleep(self.cfg["poll_interval_sec"])
                    continue

                # 6) Pauza cand ATR a fost prea mic recent
                low_atr_pause = self.cfg.get("low_atr_pause_minutes", 15)
                if self.last_low_atr_time is not None:
                    dt = now_utc - self.last_low_atr_time
                    if dt < timedelta(minutes=low_atr_pause):
                        logging.info(
                            f"ATR a fost prea mic recent (acum {dt.total_seconds()/60:.1f} min) - "
                            f"astept inainte de noi intrari."
                        )
                        time.sleep(self.cfg["poll_interval_sec"])
                        continue

                # 7) Filtru de spread / lichiditate
                if not self.spread_ok():
                    time.sleep(self.cfg["poll_interval_sec"])
                    continue

                # 8) Numar de pozitii deschise
                open_count = sum(1 for p in self.open_positions if p["status"] == "open")
                if open_count >= self.cfg["max_open_trades"]:
                    logging.info("Numarul maxim de pozitii deschise atins. Astept...")
                    time.sleep(self.cfg["poll_interval_sec"])
                    continue

                # 9) Bias HTF
                bias = self.get_htf_bias()
                logging.info(f"HTF bias: {bias}")
                if bias == "none":
                    logging.info("Fara trend clar HTF - nu iau trade-uri.")
                    time.sleep(self.cfg["poll_interval_sec"])
                    continue

                # 10) Cautam semnal pe LTF
                signal = self.liquidity_grab_signal(bias)
                if signal:
                    logging.info(f"Signal gasit: {signal}")
                    self.open_order(signal)
                else:
                    logging.info("Niciun signal valid pe LTF.")

            except Exception as e:
                logging.error(f"Eroare in loop: {e}")

            time.sleep(self.cfg["poll_interval_sec"])


if __name__ == "__main__":
    bot = ScalpingBotGateIO(CONFIG)
    bot.run()
