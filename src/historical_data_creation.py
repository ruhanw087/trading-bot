import pandas as pd
from pathlib import Path
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from datetime import datetime, timedelta
from tqdm import tqdm


DATA_DIR = Path("data")


API_KEY = "PKRYPYLJE36PG2ROIQODHW6PMK"
SECRET_KEY = "HoQ6JapAaYT8arayXiocGjyv13U2JU7uv1ruRkVo4ncM"


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
    return df

def save_parquet(symbol, df):
    path = DATA_DIR / "minute" / f"{symbol}.parquet"
    df.to_parquet(path, engine="pyarrow")


if __name__ == "__main__":
    tickers = get_russell_1000()

    for ticker in tqdm(tickers):
        try:
            df = fetch_minute_data(ticker)
            save_parquet(ticker, df)

        except Exception as e:
            print(f"Failed {ticker}", e)
