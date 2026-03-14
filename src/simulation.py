from hmm_model import load_ticker_returns, fit_returns, predict_regimes, expected_return_tomorrow
from historical_data_creation import get_russell_1000

import pandas as pd
import numpy as np
import os
import tqdm

WINDOW = 63
STARTING_CAPITAL = 10000


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

        if len(expected_returns) == 0:
            capital_history.append(capital)
            continue

        best_ticker = max(expected_returns, key=expected_returns.get)

        trade_df = ticker_data[best_ticker]["df"]

        open_price = trade_df.iloc[i+1]["open"]
        close_price = trade_df.iloc[i+1]["close"]

        shares = capital / open_price

        capital += shares * (open_price-close_price)

        capital_history.append(capital)
        ticker_history.append(best_ticker)

        print(capital)
        print(best_ticker)

    results = pd.DataFrame({
        "timestamp": dates[WINDOW+1:WINDOW+1+len(capital_history)],
        "capital": capital_history,
        "ticker": ticker_history
    })

    return results

tickers = get_russell_1000()

results = simulate_trading(tickers)

print(results)
