from hmm_model import load_ticker_returns, fit_returns, predict_regimes, expected_return_tomorrow
from historical_data_creation import get_russell_1000
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import tqdm
import warnings
import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
warnings.simplefilter("ignore", category=UserWarning)

WINDOW = 400
STARTING_CAPITAL = 500


def simulate_trading(tickers):

    ticker_data = {}

    for ticker in tickers:
        scaled_returns, df = load_ticker_returns(ticker)
        ticker_data[ticker] = {
            "returns": scaled_returns,
            "df": df.reset_index(drop=True)
        }

    dates = ticker_data[tickers[0]]["df"]["timestamp"].values

    capital = STARTING_CAPITAL
    capital_history = []
    ticker_history = []

    for i in range(WINDOW, len(dates) - 1):
        print(i)
        expected_returns = {}

        for ticker in tickers:

            data = ticker_data[ticker]
            returns = data["returns"]
            if i >= len(returns):
                continue
            if ticker_data[ticker]["df"].iloc[i]["open"]>capital:
                continue
            train_returns = returns[i-WINDOW:i]

            try:
                model = fit_returns(train_returns)

                exp_ret, _ = expected_return_tomorrow(
                    model,
                    train_returns
                )

                expected_returns[ticker] = exp_ret

            except:
                continue

        if len(expected_returns) == 0 or max(expected_returns.values())<0.002:
            print('No Buy')
            capital_history.append(capital)
            continue

        best_ticker = max(expected_returns, key=expected_returns.get)

        trade_df = ticker_data[best_ticker]["df"]

        open_price = trade_df.iloc[i]["open"]
        close_price = trade_df.iloc[i]["close"]

        shares = capital / open_price

        capital += shares * (close_price-open_price)

        capital_history.append(capital)
        ticker_history.append(best_ticker)

        print(capital)
        print(best_ticker, open_price, close_price)

    results = pd.DataFrame({
        "timestamp": dates[WINDOW+1:WINDOW+1+len(capital_history)],
        "capital": capital_history,
        "ticker": ticker_history
    })

    return results

tickers = get_russell_1000()

results = simulate_trading(tickers)

plt.plot(results['capital'])

plt.title("Portfolio Value")
plt.xlabel("Day")
plt.ylabel("Capital")

plt.show()

print(results)
