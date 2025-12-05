# ======================================================================
#                           Telegram notifier 2
# ======================================================================
"""
telegram_notifier.py

Modul simplu pentru trimitere notificari pe Telegram.
Poate fi importat si folosit in orice bot / script Python.

Exemplu de utilizare:

    from telegram_notifier import TelegramNotifier

    notifier = TelegramNotifier(
        token="TELEGRAM_BOT_TOKEN",
        chat_id=6120607540,
    )

    notifier.send("Bot pornit cu succes.")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import traceback

import requests


@dataclass
class TelegramConfig:
    """
    Config pentru TelegramNotifier.

    :param token: Tokenul botului de la BotFather (string).
    :param chat_id: chat.id al utilizatorului sau grupului (int sau string).
    :param enabled: Daca este False, nu trimite mesaje (doar logheaza la stdout).
    :param timeout: Timeout pentru request HTTP in secunde.
    """
    token: str
    chat_id: Union[int, str]
    enabled: bool = True
    timeout: int = 5


class TelegramNotifier:
    """
    Clasa pentru trimitere notificari Telegram intr-un mod safe:
      - nu arunca exceptii catre bot (doar printeaza avertismente)
      - poate fi dezactivata prin flag (enabled=False)
    """

    def __init__(self, token: str, chat_id: Union[int, str], enabled: bool = True, timeout: int = 5) -> None:
        self.config = TelegramConfig(
            token=token,
            chat_id=chat_id,
            enabled=enabled,
            timeout=timeout,
        )

        if not self.config.token or not self.config.chat_id:
            # Daca lipsesc datele, dezactivam automat
            self.config.enabled = False
            print("[TelegramNotifier] TOKEN sau CHAT_ID lipsa - dezactivez notificarea.")

    def _build_url(self) -> str:
        return f"https://api.telegram.org/bot{self.config.token}/sendMessage"

    def send(self, message: str) -> None:
        """
        Trimite un mesaj simplu pe Telegram.

        Daca enabled=False sau daca apare o eroare,
        NU ridica exceptii catre bot (doar printeaza un warning).
        """
        if not self.config.enabled:
            print(f"[TelegramNotifier OFF] {message}")
            return

        try:
            url = self._build_url()
            payload = {
                "chat_id": self.config.chat_id,
                "text": message,
            }
            requests.post(url, json=payload, timeout=self.config.timeout)
        except Exception as e:
            print(f"[TelegramNotifier WARN] Nu am putut trimite mesaj Telegram: {e}")

    def send_exception(self, prefix: str, exc: BaseException, max_length: int = 3500) -> None:
        """
        Trimite un mesaj cu traceback scurtat pentru o exceptie.

        :param prefix: Text scurt inainte de eroare (ex: 'Bot a crapat').
        :param exc: Exceptia prinsa.
        :param max_length: Lungimea maxima a textului trimis.
        """
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        text = f"{prefix}\n\n{tb}"
        if len(text) > max_length:
            text = text[:max_length] + "\n\n...[trunchiat]"

        self.send(text)
