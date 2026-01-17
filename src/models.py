# src/models.py
from __future__ import annotations

import torch
from torch import nn
from typing import Optional


# ============================================================
# Encoder-Decoder LSTM (seq2seq)
# ============================================================

class Seq2SeqLSTM(nn.Module):
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
        dropout: float = 0.2,
    ):
        super().__init__()
        if horizon is None:
            raise ValueError("horizon must be set for Seq2SeqLSTM.")

        enc_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=enc_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=enc_dropout,
        )
        self.head = nn.Linear(hidden_size, 1)
        self.horizon = int(horizon)
        self.start_token = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(x_hist)
        batch_size = x_hist.size(0)
        dec_input = self.start_token.expand(batch_size, 1, 1)

        preds = []
        h_t, c_t = h, c
        for _ in range(self.horizon):
            out, (h_t, c_t) = self.decoder(dec_input, (h_t, c_t))
            y_step = self.head(out)  # [B, 1, 1]
            preds.append(y_step.squeeze(-1))  # [B, 1]
            dec_input = y_step.detach()

        return torch.cat(preds, dim=1)  # [B, H]


# Model builder (agnostico per random search)
# ============================================================

def build_model(model_cfg, device: Optional[torch.device] = None) -> nn.Module:
    """
    Costruisce il modello in base a model_cfg.arch.
    Il random search lavora modificando model_cfg.
    """
    arch = str(getattr(model_cfg, "arch", "seq2seq")).lower()

    if arch == "seq2seq":
        model = Seq2SeqLSTM(
            input_size=model_cfg.input_size,
            horizon=model_cfg.horizon,
            hidden_size=model_cfg.seq2seq_hidden_size,
            num_layers=model_cfg.seq2seq_num_layers,
            dropout=model_cfg.seq2seq_dropout,
        )
    else:
        raise ValueError(f"Architettura non supportata: {arch}")

    if device is not None:
        model = model.to(device)

    return model
