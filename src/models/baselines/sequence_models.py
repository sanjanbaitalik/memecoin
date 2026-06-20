from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prism.evaluation.metrics import evaluate_frame


class _SequenceDataset:
    def __init__(self, features: np.ndarray, targets: np.ndarray, lookback: int):
        self.features = features
        self.targets = targets
        self.lookback = lookback

    def __len__(self) -> int:
        return max(len(self.features) - self.lookback, 0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.lookback]
        y = self.targets[idx + self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class _LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class _GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class _BiLSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class _TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.trim = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        if self.trim > 0:
            out = out[:, :, :-self.trim]
            residual = residual[:, :, :-self.trim]
        return self.relu(out + residual)


class _TCNModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 3, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        channels = [input_size] + [hidden_size] * num_layers
        for i in range(num_layers):
            layers.append(_TCNBlock(channels[i], channels[i + 1], kernel_size, dilation=2 ** i, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x.permute(0, 2, 1))
        return self.fc(out[:, :, -1]).squeeze(-1)


class _NBeatsLiteModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.lookback = input_size
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.size(0), -1)
        out = self.blocks(flat)
        return self.fc(out).squeeze(-1)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "torch_lstm": _LSTMModel,
    "torch_gru": _GRUModel,
    "torch_bilstm": _BiLSTMModel,
    "torch_tcn": _TCNModel,
    "torch_nbeats": _NBeatsLiteModel,
}


def _to_sequences(features: np.ndarray, targets: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    if len(features) <= lookback:
        return np.empty((0, lookback, features.shape[1])), np.empty(0)
    X = np.stack([features[i : i + lookback] for i in range(len(features) - lookback)])
    y = targets[lookback:]
    return X, y


def train_sequence_model(
    model_class: type[nn.Module],
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    lookback: int = 14,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_x = train[feature_cols].fillna(0.0).to_numpy(dtype=float)
    train_y = train["target_t_plus_h"].to_numpy(dtype=float)
    val_x = val[feature_cols].fillna(0.0).to_numpy(dtype=float) if not val.empty else train_x[:0]
    val_y = val["target_t_plus_h"].to_numpy(dtype=float) if not val.empty else train_y[:0]
    test_x = test[feature_cols].fillna(0.0).to_numpy(dtype=float)

    input_size = train_x.shape[1]

    if model_class is _NBeatsLiteModel:
        flat_lookback = lookback * input_size
        model = model_class(input_size=flat_lookback, hidden_size=hidden_size, num_layers=num_layers)
        X_train_seq = np.stack([train_x[i : i + lookback].ravel() for i in range(len(train_x) - lookback)]) if len(train_x) > lookback else np.empty((0, flat_lookback))
        y_train_seq = train_y[lookback:]
        X_val_seq = np.stack([val_x[i : i + lookback].ravel() for i in range(len(val_x) - lookback)]) if len(val_x) > lookback else np.empty((0, flat_lookback))
        y_val_seq = val_y[lookback:]
        X_test_seq = np.stack([test_x[i : i + lookback].ravel() for i in range(len(test_x) - lookback)]) if len(test_x) > lookback else np.empty((0, flat_lookback))
    else:
        model = model_class(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        X_train_seq, y_train_seq = _to_sequences(train_x, train_y, lookback)
        X_val_seq, y_val_seq = _to_sequences(val_x, val_y, lookback) if val_x.size > 0 else (np.empty((0, lookback, input_size)), np.empty(0))
        X_test_seq, _ = _to_sequences(test_x, np.zeros(len(test_x)), lookback)

    if X_train_seq.shape[0] == 0:
        return {"predictions": np.full(len(test), np.nan), "backend": model_name, "error": "insufficient_data"}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32))
    train_dl = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if X_val_seq.shape[0] > 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.tensor(X_val_seq, dtype=torch.float32))
                val_loss = criterion(val_pred, torch.tensor(y_val_seq, dtype=torch.float32)).item()
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        elif best_state is None:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        if X_test_seq.shape[0] > 0:
            test_pred = model(torch.tensor(X_test_seq, dtype=torch.float32)).numpy()
        else:
            test_pred = np.array([], dtype=float)

    full_pred = np.full(len(test), np.nan)
    if len(test_pred) > 0:
        full_pred[lookback : lookback + len(test_pred)] = test_pred

    return {
        "predictions": full_pred,
        "backend": model_name,
        "lookback": lookback,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "epochs_trained": min(epochs, no_improve + patience if no_improve > 0 else epochs),
    }
