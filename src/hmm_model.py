import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


n_states = 2

def load_ticker_returns(ticker):
    df = pd.read_parquet(f"data/day/{ticker}.parquet")
    returns = df["daily_return"].dropna()
    df_X = returns.values.reshape(-1, 1)
    scaler = StandardScaler()

    scaled_X = scaler.fit_transform(
        df_X
    )
    return scaled_X, df

def fit_returns(returns):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        tol=1e-3,
        random_state=0
    )
    model.fit(returns)
    return model

def predict_regimes(model, df, returns):
    hidden_states = model.predict(returns)
    df = df.iloc[1:].copy()
    df["state"] = hidden_states
    return df

def expected_return_tomorrow(model, returns):
    logprob, state_probs = model.score_samples(returns)
    current_state_dist = state_probs[-1]
    next_state_probs = current_state_dist @ model.transmat_
    state_means = model.means_.flatten()
    expected_return = next_state_probs @ state_means
    return expected_return, next_state_probs

if __name__ == "__main__":
    scaled_X, df = load_ticker_returns('A')
    model = fit_returns(scaled_X)
    df = predict_regimes(model,df,scaled_X)
    # # plt.figure(figsize=(12,6))



    # # for state in range(n_states):
    # #     mask = df["state"] == state
    # #     plt.scatter(df["timestamp"][mask], df["close"][mask], s=5, label=f"State {state}")

    # # plt.plot(df["timestamp"], df["close"], color="black", alpha=0.4)


    # # plt.legend()
    # # plt.title("HMM Market Regimes")
    # # plt.show()


    # plt.figure(figsize=(10,6))

    # for state in range(n_states):
        
    #     state_returns = df[df["state"] == state]["daily_return"]
        
    #     plt.hist(
    #         state_returns,
    #         bins=40,
    #         alpha=0.5,
    #         label=f"State {state}"
    #     )

    # plt.legend()
    # plt.title("Return Distribution by HMM State")
    # plt.xlabel("Daily Return")
    # plt.ylabel("Frequency")
    # plt.show()

