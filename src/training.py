from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


# ============================================================
# Batch helpers
# ============================================================

def _get_xy_from_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Supporta:
      - dict batch: {"x_hist": ..., "y_future": ...} (+ opzionale "x_future")
      - tuple/list: (x_hist, y_future)
    Ritorna (x_hist, y_future) come float tensors.
    """
    if isinstance(batch, dict):
        x = batch["x_hist"]
        y = batch["y_future"]
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
    else:
        raise ValueError("Batch non riconosciuto: atteso dict o tuple(len=2).")

    return x.float(), y.float()


# ============================================================
# Metriche
# ============================================================

@torch.no_grad()
def compute_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    RMSE su tutte le dimensioni (batch*horizon).
    y_true, y_pred: [B, H] o compatibili.
    """
    mse = torch.mean((y_true - y_pred) ** 2)
    return torch.sqrt(mse).item()


@torch.no_grad()
def compute_mase(y_true: torch.Tensor, y_pred: torch.Tensor, naive_scale: float) -> float:
    """
    MASE = MAE(model) / MAE(naive)
    naive_scale: scalare pre-calcolato dal train come mean(|y_t - y_{t-m}|)
    """
    mae_model = torch.mean(torch.abs(y_true - y_pred)).item()
    if naive_scale <= 0:
        return float("inf")
    return mae_model / naive_scale


@torch.no_grad()
def compute_naive_scale_from_series(
    y_insample: Union[np.ndarray, torch.Tensor],
    m: int = 24,
) -> float:
    """
    Calcola una volta sola la scala naïve per MASE:
      mean(|y_t - y_{t-m}|) su una serie insample (tipicamente il train).

    y_insample può essere:
      - numpy [N] / [N,1] / [N,H]
      - torch  [N] / [N,1] / [N,H]

    Nota: per coerenza con la loss, di solito lo fai nello stesso spazio (scaled).
    """
    if isinstance(y_insample, np.ndarray):
        y = torch.from_numpy(y_insample).float().view(-1)
    else:
        y = y_insample.detach().float().view(-1).cpu()

    if y.numel() <= m:
        return 0.0

    return torch.mean(torch.abs(y[m:] - y[:-m])).item()


# ============================================================
# Core loops
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0

    for batch in loader:
        x_hist, y_future = _get_xy_from_batch(batch)
        x_hist = x_hist.to(device)
        y_future = y_future.to(device)

        optimizer.zero_grad()
        y_hat = model(x_hist)
        loss = loss_fn(y_hat, y_future)
        loss.backward()
        optimizer.step()

        bs = x_hist.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    naive_scale: Optional[float] = None,
) -> Dict[str, float]:
    """
    Ritorna metriche medie pesate per numero di sample:
      - loss (MSE di default)
      - rmse
      - mase (se naive_scale è fornito)
    """
    model.eval()
    running_loss = 0.0
    running_rmse = 0.0
    running_mase = 0.0
    n_samples = 0

    for batch in loader:
        x_hist, y_future = _get_xy_from_batch(batch)
        x_hist = x_hist.to(device)
        y_future = y_future.to(device)

        y_hat = model(x_hist)

        loss = loss_fn(y_hat, y_future)
        rmse = compute_rmse(y_future, y_hat)

        if naive_scale is not None:
            mase = compute_mase(y_future, y_hat, naive_scale)
        else:
            mase = float("nan")

        bs = x_hist.size(0)
        running_loss += loss.item() * bs
        running_rmse += rmse * bs
        running_mase += (mase * bs) if naive_scale is not None else 0.0
        n_samples += bs

    out = {
        "loss": running_loss / max(n_samples, 1),
        "rmse": running_rmse / max(n_samples, 1),
    }
    if naive_scale is not None:
        out["mase"] = running_mase / max(n_samples, 1)
    return out


# ============================================================
# Fit wrapper (best model su val)
# ============================================================

@dataclass
class FitResult:
    model: nn.Module
    train_losses: List[float]
    val_losses: List[float]
    val_rmse: List[float]
    val_mase: List[float]


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    loss_fn: Optional[nn.Module] = None,
    # per MASE
    naive_scale: Optional[float] = None,
    # se non passi naive_scale, puoi passare y_train_insample e lo calcoliamo noi
    y_train_insample: Optional[Union[np.ndarray, torch.Tensor]] = None,
    mase_m: int = 24,
    keep_best_on_val: bool = True,
    verbose: bool = True,
) -> FitResult:
    """
    Training loop multi-epoca, con:
      - best model su validation (loss)
      - metriche val: loss, rmse, mase (se possibile)

    MASE:
      - se passi naive_scale: usiamo quello
      - altrimenti, se passi y_train_insample: calcoliamo naive_scale = mean(|y_t - y_{t-m}|)
      - altrimenti MASE non viene calcolata (NaN)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_fn or nn.MSELoss()

    # prepare naive_scale
    if naive_scale is None and y_train_insample is not None:
        naive_scale = compute_naive_scale_from_series(y_train_insample, m=mase_m)

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_rmse: List[float] = []
    val_mase: List[float] = []

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        train_losses.append(tr_loss)

        if val_loader is not None:
            metrics = evaluate_metrics(model, val_loader, device, loss_fn, naive_scale=naive_scale)
            v_loss = metrics["loss"]
            v_rmse = metrics["rmse"]
            v_mase = metrics.get("mase", float("nan"))

            val_losses.append(v_loss)
            val_rmse.append(v_rmse)
            val_mase.append(v_mase)

            if keep_best_on_val and v_loss < best_val:
                best_val = v_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if verbose:
                msg = f"Epoch {ep:02d} | Train MSE: {tr_loss:.4f} | Val MSE: {v_loss:.4f} | Val RMSE: {v_rmse:.4f}"
                if naive_scale is not None:
                    msg += f" | Val MASE: {v_mase:.4f}"
                print(msg)
        else:
            if verbose:
                print(f"Epoch {ep:02d} | Train MSE: {tr_loss:.4f}")

    if keep_best_on_val and best_state is not None:
        model.load_state_dict(best_state)

    return FitResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        val_rmse=val_rmse,
        val_mase=val_mase,
    )

@torch.no_grad()
def predict_over_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ritorna (y_true_np, y_pred_np) concatenati su tutti i batch.
    """
    model.eval()
    model.to(device)

    ys: List[np.ndarray] = []
    yhs: List[np.ndarray] = []

    for batch in loader:
        x_hist, y_future = _get_xy_from_batch(batch)
        x_hist = x_hist.to(device)

        y_hat = model(x_hist)

        ys.append(y_future.detach().cpu().numpy())
        yhs.append(y_hat.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0) if ys else np.empty((0,))
    y_pred = np.concatenate(yhs, axis=0) if yhs else np.empty((0,))
    return y_true, y_pred

@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Compatibilità con il tuo evaluate.py attuale:
      mse_scaled = evaluate_loss(model, val_loader, nn.MSELoss(), device)
    """
    metrics = evaluate_metrics(model, loader, device, criterion, naive_scale=None)
    return float(metrics["loss"])