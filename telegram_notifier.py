import os
import traceback
import requests
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _env_bool(name, default=False):
    v = os.getenv(name)
    if not v:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "y")

@dataclass
class TelegramConfig:
    token: str
    chat_id: str
    enabled: bool = True
    timeout: int = 5

class TelegramNotifier:
    def __init__(self):
        token = os.getenv("TELEGRAM_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        enabled = _env_bool("TELEGRAM_ENABLED", True)
        timeout = int(os.getenv("TELEGRAM_TIMEOUT", "5"))

        if not token or not chat_id:
            enabled = False

        self.config = TelegramConfig(
            token=token,
            chat_id=chat_id,
            enabled=enabled,
            timeout=timeout
        )

    def _base(self):
        return f"https://api.telegram.org/bot{self.config.token}"

    def send(self, text):
        if not self.config.enabled:
            print("[TG OFF]", text)
            return
        try:
            url = self._base() + "/sendMessage"
            requests.post(url, json={"chat_id": self.config.chat_id, "text": text}, timeout=self.config.timeout)
        except Exception as e:
            print(f"[TG] Error sending message: {e}")

    def send_exception(self, prefix, exc):
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        msg = f"{prefix}\n\n{tb}"
        if len(msg) > 3500:
            msg = msg[:3500] + "\n...[cut]..."
        self.send(msg)

    def send_file(self, filepath, caption=""):
        if not self.config.enabled:
            return
        try:
            url = self._base() + "/sendDocument"
            with open(filepath, "rb") as f:
                requests.post(
                    url,
                    data={"chat_id": self.config.chat_id, "caption": caption},
                    files={"document": f},
                    timeout=20
                )
        except Exception as e:
            print(f"[TG] send_file error:", e)

    def get_updates(self, offset=None):
        if not self.config.enabled:
            return []

        try:
            url = self._base() + "/getUpdates"
            params = {"timeout": 1, "limit": 20}
            if offset:
                params["offset"] = offset
            resp = requests.get(url, params=params, timeout=5).json()
            if not resp.get("ok"):
                return []
            return resp.get("result", [])
        except:
            return []
