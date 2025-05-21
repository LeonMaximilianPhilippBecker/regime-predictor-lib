import yfinance as yf

yf.enable_debug_mode()
stock_name = "AAPL"
start_date = "2020-01-01"
end_date = "2020-01-10"
data = yf.download(stock_name, start=start_date, end=end_date)
data
