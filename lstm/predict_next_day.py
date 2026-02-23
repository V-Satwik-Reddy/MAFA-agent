"""Utility to load a trained LSTM model for a ticker and predict the next-day close
from the most recent ~1 month (>=20 trading days) of OHLCV data.

Usage (example):

>>> import pandas as pd
>>> from agent_tools.predict_next_day import predict_next_day_price
>>> df = pd.read_csv("/path/to/last_month_prices.csv")
>>> pred = predict_next_day_price("AAPL", df)
>>> print(pred)

Requires that you have already trained and saved the model under
output/{TICKER}/ with model.keras and scaler.joblib.
"""
import os
import tempfile
from typing import Union

import pandas as pd

from lstm.infer import ModelWrapper


def predict_next_day_price(ticker: str, recent_prices: pd.DataFrame) -> float:
    """Predict the next-day close for ``ticker`` using its saved LSTM model.

    Args:
        ticker: Ticker symbol, must have a model under output/{ticker}/.
        recent_prices: DataFrame with at least 21 rows of recent daily OHLCV data
            (columns should include date + open, high, low, close, volume). The most
            recent row should be the latest available trading day; the model will use
            the last 20 rows as the input window and predict the following day.

    Returns:
        The predicted next-day close price as a float.
    """
    if len(recent_prices) < 21:
        raise ValueError("Need at least 21 rows (20 for input window, 1 target) of recent prices")

    model_dir = os.path.join("lstm", "output", ticker)
    if not os.path.exists(os.path.join(model_dir, "model.keras")):
        raise FileNotFoundError(f"model.keras not found under {model_dir}; train the model first")

    # Save to a temporary CSV because ModelWrapper expects a file path.
    # On Windows, keep delete=False and remove manually to avoid permission issues when re-opening.
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8")
    tmp_path = tmp.name
    try:
        recent_prices.to_csv(tmp_path, index=False)
        tmp.close()
        wrapper = ModelWrapper(model_dir=model_dir)
        df_pred = wrapper.predict_from_csv(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if df_pred.empty:
        raise RuntimeError("Prediction output is empty")

    # The last row corresponds to the most recent input window.
    return float(df_pred.iloc[-1]["y_pred"])
