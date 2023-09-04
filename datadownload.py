import yfinance as yf
import pandas as pd
from datetime import datetime

# Download Google stock data
ticker = "ETH-USD"

start_date = "2015-09-02"
# Get the current date as the end date
#end_date = datetime.today().strftime('%Y-%m-%d') #today date
end_date = "2022-09-02"

data = yf.download(ticker, start=start_date, end=end_date)

# Format the data
data = data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(inplace=True)

# Save to CSV in the specified folder
folder = r"C:\\Users\\dj_m0\\Documents\\PythonScripts\\Trading-Agent\\data"
csv_file = folder + "\\" + ticker + "_" + start_date +"_" + end_date + ".csv"
data.to_csv(csv_file, index=False)

print("Data saved to", csv_file)
