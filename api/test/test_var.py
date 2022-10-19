import pandas_datareader as pdr
import pandas as pd
import numpy as np
from pandas_datareader.yahoo.daily import YahooDailyReader
import datetime as dt
from sklearn.decomposition import PCA
from typing import List, Dict


def test_var(
    symbols: List[str],
    exposures: Dict[str, float],
    start: dt.date,
    end: dt.date,
    pca_components: int = 5,
):
    """Uses principal components analysis to calculate the 99% 10 day VaR with historical daily stock data.

    Args:
        symbols: Stock ticker symbols to pull daily data for PCA.
        exposures: Mapping to symbol to $ PnL given a 1 basis point move in the stock.
        start: Start date to pull daily stock data.
        end: End date to pull daily stock data.
    """
    # Fetch data
    reader = YahooDailyReader(symbols=symbols, start=start, end=end)
    price_df = reader.read()

    # Compute daily returns
    return_df = pd.DataFrame()
    for symbol in symbols:
        return_df[symbol] = np.log(
            price_df[("Close", symbol)] / price_df[("Open", symbol)]
        )
    cov = np.cov(return_df)
    corr = np.corrcoef(return_df)

    # Compute principal components
    pca = PCA(n_components=pca_components)
    pca.fit(return_df)

    total_explained_variance = sum(pca.explained_variance_ratio_)
    # Compute exposures to each factor ($ per factor score % change)
    # ie a one pct move in the factor / component causes $x portfolio value change
    exposure_vector = np.array([exposures.get(symbol, 0) for symbol in symbols])

    # Factor exposures = F x E
    factor_exposures = np.dot(pca.components_, exposure_vector)

    # Portfolio SD = sqrt(Factor exposures ** 2 x Factor SD ** 2)
    portfolio_sd = np.sqrt(np.dot(factor_exposures ** 2, pca.explained_variance_ ** 2))
    value_at_risk = np.sqrt(10) * portfolio_sd * 2.326
    return value_at_risk


if __name__ == "__main__":
    portfolio_1_var = test_var(
        symbols=["F", "FB", "AMZN", "AAPL", "GOOG", "MSFT", "JPM", "BAC"],
        exposures={"F": 10, "FB": 10, "AAPL": 10, "JPM": 10, "BAC": 10},
        start=dt.date(2020, 1, 1),
        end=dt.date(2022, 5, 1),
    )

    portfolio_2_var = test_var(
        symbols=["F", "FB", "AMZN", "AAPL", "GOOG", "MSFT", "JPM", "BAC"],
        exposures={"FB": 10, "AAPL": 10, "GOOG": 10, "MSFT": 10},
        start=dt.date(2020, 1, 1),
        end=dt.date(2022, 5, 1),
    )

    portfolio_3_var = test_var(
        symbols=["F", "FB", "AMZN", "AAPL", "GOOG", "MSFT", "JPM", "BAC"],
        exposures={"FB": 10, "AAPL": -10, "GOOG": 10, "MSFT": 10},
        start=dt.date(2020, 1, 1),
        end=dt.date(2022, 5, 1),
    )

    portfolio_4_var = test_var(
        symbols=["F", "FB", "AMZN", "AAPL", "GOOG", "MSFT", "JPM", "BAC"],
        exposures={"FB": 10, "AAPL": 10, "GOOG": 10, "MSFT": 10, "JPM": 10},
        start=dt.date(2020, 1, 1),
        end=dt.date(2022, 5, 1),
    )

    print(f"{portfolio_1_var=}")
    print(f"{portfolio_2_var=}")
    print(f"{portfolio_3_var=}")
    print(f"{portfolio_4_var=}")
