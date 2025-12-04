# src/models.py

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Type

import torch
import torch.nn as nn

ModelName = Literal["lstm", "gru", "convlstm"]


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

class ConvLSTMModel(nn.Module):
    """
    Conv1D + LSTM many-to-one:
    x [B, T, F] -> conv1d lungo T -> LSTM -> y_hat [B, horizon]
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

        # Conv1D: canali = feature, asse spaziale = tempo
        # input:  [B, F, T]
        # output: [B, hidden_size, T]
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()

        # LSTM lavora sulla sequenza filtrata [B, T, hidden_size]
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T] per Conv1d
        x = x.transpose(1, 2)

        # Conv1D + BN + ReLU
        x = self.conv1(x)          # [B, hidden_size, T]
        x = self.bn1(x)
        x = self.relu(x)

        # Torniamo a [B, T, H] per l'LSTM
        x = x.transpose(1, 2)      # [B, T, hidden_size]

        batch_size = x.size(0)
        h0 = x.new_zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = x.new_zeros(self.num_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))   # [B, T, H]
        last_hidden = out[:, -1, :]       # [B, H]
        last_hidden = self.dropout(last_hidden)
        y_hat = self.fc(last_hidden)      # [B, horizon]

        return y_hat

# ============================================================
# ConvLSTM "2D": Conv2d sul tempo + LSTM + Linear
# ============================================================

class ConvLSTM2DModel(nn.Module):
    """
    Conv2d su tensore [B, 1, T, F] per estrarre pattern temporali 2D,
    poi pooling sulle feature, poi LSTM many-to-one, poi Linear sull'horizon.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size

        # Conv2d: in_channels=1 (singolo "canale" temporale),
        # out_channels=hidden_size (numero di filtri),
        # kernel_size=(3,1): guarda 3 timestep alla volta, non mescola le feature.
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=hidden_size,
                kernel_size=(3, 1),
                padding=(1, 0),    # manteniamo stessa T
            ),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        # LSTM che lavora sulla sequenza filtrata
        # Dopo la conv + pooling avremo feature_dim = hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        steps:
          - reshape a [B, 1, T, F]
          - conv2d -> [B, C, T, F]
          - mean pooling sulle feature (dim=3) -> [B, C, T]
          - permuta a [B, T, C] per LSTM
          - LSTM many-to-one
          - Linear -> [B, horizon]
        """
        B, T, F = x.shape

        # [B, 1, T, F]
        x_2d = x.unsqueeze(1)

        # [B, C, T, F]
        x_conv = self.conv2d(x_2d)

        # pooling sulle feature: [B, C, T]
        x_pooled = x_conv.mean(dim=3)

        # permutiamo a [B, T, C] per LSTM
        x_seq = x_pooled.permute(0, 2, 1)

        # LSTM
        out, _ = self.lstm(x_seq)        # [B, T, H]
        last_hidden = out[:, -1, :]      # [B, H]

        # output horizon
        y_hat = self.fc(last_hidden)     # [B, horizon]
        return y_hat


_MODEL_REGISTRY: Dict[ModelName, Type[nn.Module]] = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "convlstm": ConvLSTMModel,
    "convlstm2d": ConvLSTM2DModel
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
