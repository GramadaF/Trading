import requests
import traceback

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, enabled: bool = True, timeout: int = 5):
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled
        self.timeout = timeout

    def send(self, message: str):
        if not self.enabled:
            return

        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message
            }
            requests.post(url, data=data, timeout=self.timeout)
        except Exception:
            print("Eroare trimitere notificare Telegram:", traceback.format_exc())

    def send_exception(self, prefix: str, exc: BaseException):
        msg = f"{prefix}\n{type(exc).__name__}: {exc}"
        self.send(msg)

    def get_updates(self, offset=None, limit=20):
        if not self.enabled:
            return []

        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params = {
                "offset": offset,
                "limit": limit
            }
            resp = requests.get(url, params=params, timeout=self.timeout)
            data = resp.json()
            return data.get("result", [])
        except Exception:
            print("Eroare get_updates Telegram:", traceback.format_exc())
            return []
