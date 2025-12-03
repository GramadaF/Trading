import ccxt

API_KEY = "s744AL2Y6htP5VegHLdL4pcSgXwfMiSX7z3RbxlD81NTltqBzw9NtcmKJ2ZDgG6r"  # pune cheile tale aici
API_SECRET = "hRg5RGEf1wvMjFjxLMAVcAF0QODfWbJmMiwtZBby5WoDMejic8AivkXV0v0VnMYG"

exchange = ccxt.binance({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": "future"}
})

exchange.set_sandbox_mode(False)  # sau True, dar să corespundă cu tipul cheilor

try:
    print("Loading markets...")
    exchange.load_markets()
    print("Markets loaded OK.")

    balance = exchange.fetch_balance()
    print("Balance fetched OK.")
    print(balance["USDT"])  # să vezi măcar ceva la USDT

except Exception as e:
    print("Error:", e)
