# src/training.py

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 5          # nuovi
    min_delta: float = 0.0     # miglioramento minimo per dire "ok, è meglio"


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        x = batch["x_hist"].to(device)       # [B, T, F]
        y = batch["y_future"].to(device)     # [B, horizon]

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x.size(0)

    return epoch_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    epoch_loss = 0.0

    for batch in dataloader:
        x = batch["x_hist"].to(device)
        y = batch["y_future"].to(device)

        y_hat = model(x)
        loss = criterion(y_hat, y)

        epoch_loss += loss.item() * x.size(0)

    return epoch_loss / len(dataloader.dataset)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: TrainingConfig,
    device: torch.device,
) -> Dict[str, Any]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history = {"train_loss": [], "val_loss": []}

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improve = 0

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        msg = f"[Epoch {epoch+1}/{config.epochs}] train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f" val_loss={val_loss:.4f}"
        print(msg)

        # --- early stopping ---
        if val_loss is not None:
            if val_loss + config.min_delta < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= config.patience:
                print(
                    f"[EarlyStopping] Nessun miglioramento su val per "
                    f"{config.patience} epoche. Stop a epoch {epoch+1}."
                )
                break

    # ripristina i pesi migliori trovati
    if best_state is not None:
        model.load_state_dict(best_state)

    return {"model": model, "history": history}
