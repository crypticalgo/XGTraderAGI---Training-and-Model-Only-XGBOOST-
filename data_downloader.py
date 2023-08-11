from binance.client import Client
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

# replace YOUR_API_KEY and YOUR_SECRET_KEY with your own API key and secret key
client = Client(api_key=os.environ.get('BINANCE_APIKEY'), api_secret=os.environ.get('BINANCE_SCAPIKEY'))

# define the symbol and the timeframe
symbol = 'XRPUSDT'
timeframe = Client.KLINE_INTERVAL_5MINUTE

now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

# define the start and end date in milliseconds
start_date = pd.Timestamp('2023-07-29 00:00:00')
end_date = pd.Timestamp(now)

# initialize an empty list to store the data
data_list = []

# loop over each day between the start and end dates
current_date = start_date
while current_date <= end_date:
    # convert the current date to milliseconds
    current_timestamp = int(current_date.timestamp() * 1000)
    
    # download the klines for the current day
    try:
        klines = client.futures_klines(symbol=symbol, interval=timeframe, startTime=current_timestamp, endTime=current_timestamp+86400000-1)
    except Exception as e:
        print("Error downloading data for {}: {}".format(current_date.date(), str(e)))
        current_date += pd.Timedelta(days=1)
        continue
    
    # convert the klines to a Pandas DataFrame
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    # convert the timestamp to a datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # set the timestamp as the index of the DataFrame
    df.set_index('timestamp', inplace=True)
    
    # select only the columns you want to keep
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # capitalize the first letter of column names
    df.columns = [col.capitalize() for col in df.columns]
    
    # append the DataFrame to the data list
    data_list.append(df)
    
    # move to the next day
    current_date += pd.Timedelta(days=1)
    print("Downloading data for {}...".format(current_date.date()))

    # sleep for a short time to avoid hitting API rate limits
    time.sleep(0.5)

# concatenate all the DataFrames into a single DataFrame
df = pd.concat(data_list)

# save the data to a CSV file
df.to_csv('./data/'+symbol+'_'+timeframe+'.csv', mode='w', header=True)


#############################################################################################################################################################################

# from binance.client import Client
# import pandas as pd
# import time
# import os
# from dotenv import load_dotenv

# load_dotenv()

# client = Client(api_key=os.environ.get('BINANCE_APIKEY'), api_secret=os.environ.get('BINANCE_SCAPIKEY'))

# timeframe = Client.KLINE_INTERVAL_3MINUTE

# now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
# start_date = pd.Timestamp('2020-01-01 00:00:00')
# end_date = pd.Timestamp(now)

# # symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'LTCUSDT']

# symbols = ['XRPUSDT', 'LTCUSDT']

# for symbol in symbols:
#     data_list = []
#     current_date = start_date
#     while current_date <= end_date:
#         current_timestamp = int(current_date.timestamp() * 1000)
#         try:
#             klines = client.futures_klines(symbol=symbol, interval=timeframe, startTime=current_timestamp, endTime=current_timestamp+86400000-1)
#         except Exception as e:
#             print("Error downloading data for {}: {}".format(current_date.date(), str(e)))
#             current_date += pd.Timedelta(days=1)
#             continue
        
#         df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
#         df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#         df.set_index('timestamp', inplace=True)
#         df = df[['open', 'high', 'low', 'close', 'volume']]
#         df.columns = [col.capitalize() for col in df.columns]
#         data_list.append(df)
#         current_date += pd.Timedelta(days=1)
#         print("Downloading data for {}...".format(current_date.date()))
#         time.sleep(0.5)

#     df = pd.concat(data_list)
#     df.to_csv('./data/'+symbol+'_'+timeframe+'.csv', mode='w', header=True)
