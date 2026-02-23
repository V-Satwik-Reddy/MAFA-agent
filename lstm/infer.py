"""Inference utilities for saved LSTM models.

Provides `ModelWrapper` which loads a saved Keras model, scaler, and
feature metadata from a model directory (e.g., ``output/AAPL``) and offers a
convenient ``predict_from_csv`` helper.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

DEFAULT_FEATURES: List[str] = ["open", "high", "low", "close", "volume"]


class ModelWrapper:
    """Load a trained model and run sliding-window predictions."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "model.keras")
        self.scaler_path = os.path.join(model_dir, "scaler.joblib")
        self.metrics_path = os.path.join(model_dir, "metrics.json")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Missing model file at {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Missing scaler file at {self.scaler_path}")

        # Load artifacts.
        self.model = tf.keras.models.load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        # Infer window size from the model input shape (batch, timesteps, features).
        self.window_size: int = int(self.model.input_shape[1])

        # Feature metadata comes from metrics.json when available.
        self.features: List[str] = DEFAULT_FEATURES
        self.target_col: Optional[str] = None
        self.target_idx: Optional[int] = None
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            feats = meta.get("features")
            if feats:
                self.features = list(feats)
            target_idx = meta.get("target_idx")
            if isinstance(target_idx, int) and 0 <= target_idx < len(self.features):
                self.target_idx = target_idx
                self.target_col = self.features[target_idx]

    def predict_from_csv(self, csv_path: Union[str, os.PathLike]) -> pd.DataFrame:
        """Run inference over a CSV of OHLCV data.

        The CSV must contain the feature columns used during training. If a
        ``date`` column is present, it will be carried through to the output so
        that predictions can be aligned to specific days.
        """
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("Input CSV is empty")

        if "date" in df.columns:
            df = df.sort_values("date")

        missing = [col for col in self.features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        feature_values = df[self.features].astype(float)
        scaled = self.scaler.transform(feature_values)

        if len(scaled) <= self.window_size:
            raise ValueError(
                f"Need at least {self.window_size + 1} rows to create one window; "
                f"got {len(scaled)}"
            )

        windows = []
        for idx in range(self.window_size, len(scaled)):
            windows.append(scaled[idx - self.window_size : idx])
        x_input = np.asarray(windows, dtype=np.float32)

        preds_scaled = self.model.predict(x_input, verbose=0).flatten()

        if self.target_idx is not None:
            # Inverse-scale predictions back to the original close-price space.
            placeholder = np.zeros((len(preds_scaled), len(self.features)), dtype=np.float32)
            placeholder[:, self.target_idx] = preds_scaled
            preds_unscaled = self.scaler.inverse_transform(placeholder)[:, self.target_idx]
            preds = preds_unscaled
        else:
            preds = preds_scaled

        start = self.window_size
        output = pd.DataFrame({"y_pred": preds})

        if "date" in df.columns:
            output.insert(0, "date", df.iloc[start:]["date"].values)
        if self.target_col and self.target_col in df.columns:
            output.insert(1, "y_true", df.iloc[start:][self.target_col].values)

        return output
