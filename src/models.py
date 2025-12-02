# src/models.py

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Type

import torch
import torch.nn as nn

ModelName = Literal["lstm", "gru"]


@dataclass
class ModelConfig:
    model_name: ModelName = "lstm"
    input_size: int = 1          # sovrascritto a runtime
    horizon: int = 24            # = PVDataConfig.horizon_hours
    hidden_size: int = 64
    num_layers: int = 3
    dropout: float = 0.2


class LSTMModel(nn.Module):
    """
    Many-to-one LSTM: x [B, T, F] -> y_hat [B, horizon]
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        horizon: int = 24,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = x.new_zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = x.new_zeros(self.num_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))   # [B, T, H]
        last_hidden = out[:, -1, :]       # [B, H]
        y_hat = self.fc(last_hidden)      # [B, horizon]
        return y_hat


class GRUModel(nn.Module):
    """
    Many-to-one GRU: x [B, T, F] -> y_hat [B, horizon]
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        horizon: int = 24,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = x.new_zeros(self.num_layers, batch_size, self.hidden_size)

        out, _ = self.gru(x, h0)         # [B, T, H]
        last_hidden = out[:, -1, :]      # [B, H]
        y_hat = self.fc(last_hidden)     # [B, horizon]
        return y_hat


_MODEL_REGISTRY: Dict[ModelName, Type[nn.Module]] = {
    "lstm": LSTMModel,
    "gru": GRUModel,
}


def build_model(config: ModelConfig, device: Optional[torch.device] = None) -> nn.Module:
    model_cls = _MODEL_REGISTRY[config.model_name]
    model = model_cls(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        horizon=config.horizon,
        dropout=config.dropout,
    )
    if device is not None:
        model = model.to(device)
    return model
