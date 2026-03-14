import pandas as pd
from pathlib import Path
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from datetime import datetime, timedelta
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()


DATA_DIR = Path("data")


API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')


client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def get_russell_1000():
    universe_path = DATA_DIR / "Universe.csv"
    df = pd.read_csv(universe_path)
    tickers = df['Symbol'].tolist()
    return tickers

def fetch_minute_data(symbol):

    end = datetime.utcnow()
    start = end - timedelta(days=60)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.IEX
    )
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    path = DATA_DIR / "minute" / f"{symbol}.parquet"
    df.to_parquet(path, engine="pyarrow")
    return df

def fetch_daily_data(symbol):

    end = datetime.utcnow()
    start = end - timedelta(days=365)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX
    )
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    df['daily_return'] = df['close'].pct_change()
    path = DATA_DIR / "day" / f"{symbol}.parquet"
    df.to_parquet(path, engine="pyarrow")
    return df

# def save_parquet(symbol, df):
#     path = DATA_DIR / "minute" / f"{symbol}.parquet"
#     df.to_parquet(path, engine="pyarrow")


if __name__ == "__main__":
    # tickers = get_russell_1000()

    # for ticker in tqdm(tickers):
    #     try:
    #         #df = fetch_minute_data(ticker)
    #         df = fetch_daily_data(ticker)

    #     except Exception as e:
    #         print(f"Failed {ticker}", e)

    df = pd.read_parquet('data/day/A.parquet', engine = 'pyarrow')
    df.to_csv('output.csv')
