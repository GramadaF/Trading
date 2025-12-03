import ccxt

exchange = ccxt.binance({
    "enableRateLimit": True,
})

try:
    print("Loading public markets...")
    exchange.load_markets()
    print("Public markets loaded OK. Num symbols:", len(exchange.markets))
except Exception as e:
    print("Error:", e)
