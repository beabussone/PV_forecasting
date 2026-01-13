# src/models.py
from __future__ import annotations

import torch
from torch import nn
from typing import Optional


# ============================================================
# Temporal Convolutional Network
# ============================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Input:  x_hist [B, T, F]
    Output: y_hat  [B, horizon]
    """
    def __init__(
        self,
        input_size: int,
        horizon: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels[-1], horizon)

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        # x_hist: [B, T, F] → [B, F, T]
        x = x_hist.transpose(1, 2)
        y = self.tcn(x)
        y_last = y[:, :, -1]        # last time step
        return self.head(y_last)    # [B, horizon]


# ============================================================
# LSTM + MLP
# ============================================================

class LSTMMLP(nn.Module):
    """
    Input:  x_hist [B, T, F]
    Output: y_hat  [B, horizon]
    """
    def __init__(
        self,
        input_size: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = False,
        mlp_hidden_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )

        lstm_out_size = hidden_size * (2 if bidirectional else 1)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_out_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, horizon),
        )

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x_hist)
        y_last = y[:, -1, :]
        return self.mlp(y_last)


# ============================================================
# Model builder (agnostico per random search)
# ============================================================

def build_model(model_cfg, device: Optional[torch.device] = None) -> nn.Module:
    """
    Costruisce il modello in base a model_cfg.arch.
    Il random search lavora modificando model_cfg.
    """
    # modifica per tuning
    arch = str(getattr(model_cfg, "arch", "tcn")).lower()

    if arch == "tcn":
        if getattr(model_cfg, "num_channels", None):
            num_channels = list(model_cfg.num_channels)
        else:
            hidden = int(getattr(model_cfg, "hidden", 64))
            n_blocks = int(getattr(model_cfg, "n_blocks", 3))
            num_channels = [hidden] * max(n_blocks, 1)

        model = TCN(
            input_size=model_cfg.input_size,
            horizon=model_cfg.horizon,
            num_channels=num_channels,
            kernel_size=model_cfg.kernel_size,
            dropout=model_cfg.dropout,
        )
    elif arch == "lstm_mlp":
        model = LSTMMLP(
            input_size=model_cfg.input_size,
            horizon=model_cfg.horizon,
            hidden_size=model_cfg.hidden_size,
            num_layers=model_cfg.num_layers,
            bidirectional=model_cfg.bidirectional,
            mlp_hidden_size=model_cfg.mlp_hidden_size,
            dropout=model_cfg.dropout,
        )
    else:
        raise ValueError(f"Architettura non supportata: {arch}")

    if device is not None:
        model = model.to(device)

    return model
