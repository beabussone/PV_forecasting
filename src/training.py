import torch
from torch import nn
import numpy as np


def _get_xy_from_batch(batch):
    """
    Support both dict batches (from PVForecastDataset) and tuple batches.
    Returns float tensors (x_hist, y_future).
    """
    if isinstance(batch, dict):
        x = batch["x_hist"]
        y = batch["y_future"]
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
    else:
        raise ValueError("Batch non riconosciuto: atteso dict o tuple(len=2).")
    return x.float(), y.float()

def train_one_model(model, train_loader, eval_loader=None, epochs=30, lr=1e-3, device=torch.device("cpu")):
    """
    Train a model for forecasting using MSE loss and Adam optimizer.
    Returns: (trained_model, train_losses, eval_losses)
    """
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses, eval_losses = [], []

    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        ep_train_loss = 0.0
        for batch in train_loader:
            x, y = _get_xy_from_batch(batch)
            x = x.to(device)                   # (B, L, D)
            y = y.to(device)                   # (B, horizon)

            optim.zero_grad()
            y_hat = model(x)                   # (B, horizon)
            loss = loss_fn(y_hat, y)           # scalar
            loss.backward()
            optim.step()

            ep_train_loss += loss.item() * x.size(0)

        ep_train_loss /= len(train_loader.dataset)
        train_losses.append(ep_train_loss)

        if eval_loader is not None:
            # ---- Eval ----
            model.eval()
            ep_eval_loss = 0.0
            with torch.no_grad():
                for batch in eval_loader:
                    x, y = _get_xy_from_batch(batch)
                    x = x.to(device)
                    y = y.to(device)
                    y_hat = model(x)
                    loss = loss_fn(y_hat, y)
                    ep_eval_loss += loss.item() * x.size(0)
            ep_eval_loss /= len(eval_loader.dataset)
            eval_losses.append(ep_eval_loss)
            print(f"Epoch {ep:02d} | Train MSE: {ep_train_loss:.4f} | Eval MSE: {ep_eval_loss:.4f}")
        else:
            print(f"Epoch {ep:02d} | Train MSE: {ep_train_loss:.4f}")

    return model, train_losses, eval_losses


def evaluate(model, loader, criterion, device=torch.device("cpu")):
    """
    Calcola la loss media (MSE) su un DataLoader.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x, y = _get_xy_from_batch(batch)
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def fit(model, train_loader, val_loader=None, config=None, device=torch.device("cpu")):
    """
    Training loop con tracciamento delle loss e restituzione del best model (val).
    """
    if config is None:
        raise ValueError("config di training mancante (epochs, lr, ...).")

    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_state = None
    best_val = float("inf")

    for ep in range(1, config.epochs + 1):
        model.train()
        ep_train_loss = 0.0

        for batch in train_loader:
            x, y = _get_xy_from_batch(batch)
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optim.step()

            ep_train_loss += loss.item() * x.size(0)

        ep_train_loss /= len(train_loader.dataset)
        train_losses.append(ep_train_loss)

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                # salvo copia su CPU per sicurezza
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        msg = f"Epoch {ep:02d} | Train MSE: {ep_train_loss:.4f}"
        if val_loss is not None:
            msg += f" | Val MSE: {val_loss:.4f}"
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def predict_over_loader(model, loader, device=torch.device("cpu")):
    """
    Run the model on a DataLoader and collect predictions (y_hat) and targets (y_true).
    Returns: (y_true_np, y_pred_np)
    """
    model.eval()
    ys, yhs = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = _get_xy_from_batch(batch)
            x = x.to(device)
            y_hat = model(x)
            ys.append(y.cpu().numpy())
            yhs.append(y_hat.detach().cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(yhs, axis=0)
    return y_true, y_pred
