# evaluate.py

#GLI ADATTAMENTI PER RUNNARE QUESTO FILE SONO STATI FATTI SOLO IN RELAZIONE ALLO SPLIT TEMPORALE SEMPLICE NON LA CV

# src/evaluate.py

# GLI ADATTAMENTI PER RUNNARE QUESTO FILE SONO STATI FATTI SOLO IN RELAZIONE
# ALLO SPLIT TEMPORALE SEMPLICE (NON CV)

import torch
import numpy as np
import pickle
import pandas as pd

from src.data_module import PVForecastDataset, build_dataloader
from src.models import build_model
from src.config import ExperimentConfig
from src.training import evaluate as eval_loss


def inverse_scale_y(y_scaled, scaler):
    """
    Inverse transform del target usando le statistiche in scaler["y_stats"].
    Accetta array 1D o 2D, ritorna stessa shape in spazio reale (kW/kWp).
    """
    mode = scaler["mode"]
    stats = scaler["y_stats"]

    y_scaled = np.asarray(y_scaled, dtype=float)

    if mode == "standard":
        return y_scaled * stats["std"] + stats["mean"]
    else:
        return y_scaled * (stats["max"] - stats["min"]) + stats["min"]


def compute_metrics(y_true, y_pred):
    """
    Calcola MAE, MSE, RMSE tra y_true e y_pred (stesso spazio).
    y_true, y_pred: array compatibili (1D o 2D).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    return mae, mse, rmse


def mase(y_true, y_pred, insample, m: int = 1) -> float:
    """
    MASE (Mean Absolute Scaled Error).

    y_true:   array out-of-sample (test) [N] o [N, H]
    y_pred:   array predizioni modello, stessa shape di y_true
    insample: serie storica "insample" (tipicamente il train) [T] o [T, ...]
              usata per il calcolo dell'errore del naïve.
    m:        periodo stagionale (es. 1 per naïve semplice, 24 per dati orari).

    Ritorna:
        valore scalare della MASE.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    insample = np.asarray(insample, dtype=float).ravel()

    if len(insample) <= m:
        raise ValueError("La serie insample deve essere più lunga di m.")

    # MAE del modello
    mae_model = np.mean(np.abs(y_true - y_pred))

    # MAE del naïve stagionale: y_hat_t = y_{t-m}
    naive_forecast = insample[m:]
    naive_prev = insample[:-m]
    mae_naive = np.mean(np.abs(naive_forecast - naive_prev))

    return mae_model / mae_naive


def main():
    print("=== Evaluate Test Set ===")

    cfg = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------
    # 1. Carico scaler salvato
    # -------------------------------------------------------
    scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))

    # -------------------------------------------------------
    # 2. Carico i dataset salvati dal main (già FE + scaling)
    # -------------------------------------------------------
    X_test_arr = np.load("artifacts/X_test_scaled.npy")
    y_test_arr = np.load("artifacts/y_test_scaled.npy")

    # Converto in DataFrame per soddisfare PVForecastDataset
    X_test = pd.DataFrame(X_test_arr)

    # y_test probabilmente è shape (N, 1) – ci assicuriamo che sia 2D
    if y_test_arr.ndim == 1:
        y_test_arr = y_test_arr.reshape(-1, 1)
    y_test = pd.DataFrame(y_test_arr, columns=["kwp"])  # nome colonna coerente

    test_ds = PVForecastDataset(
        X_test,
        y_test,
        cfg.data,
    )

    # LOADER
    test_loader = build_dataloader(
        test_ds,
        batch_size=64,
        shuffle=False,
    )

    # -------------------------------------------------------
    # 3. Ricostruisco modello e carico pesi
    # -------------------------------------------------------
    cfg.model.input_size = test_ds.X_values.shape[1]
    cfg.model.horizon = test_ds.horizon

    model = build_model(cfg.model, device)
    model.load_state_dict(torch.load("artifacts/model_gru.pth", map_location=device))
    model.eval()

    # -------------------------------------------------------
    # 4. Metriche nello spazio scalato
    # -------------------------------------------------------
    criterion = torch.nn.MSELoss()
    mse_scaled = eval_loss(model, test_loader, criterion, device)
    rmse_scaled = float(np.sqrt(mse_scaled))

    # -------------------------------------------------------
    # 5. Predizioni + inverse scaling (test)
    # -------------------------------------------------------
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x_hist"].to(device)
            y = batch["y_future"].cpu().numpy()
            y_hat = model(x).cpu().numpy()

            y_true_list.append(y)
            y_pred_list.append(y_hat)

    y_true_scaled = np.concatenate(y_true_list, axis=0)   # [N, H]
    y_pred_scaled = np.concatenate(y_pred_list, axis=0)   # [N, H]

    # inverse scaling in spazio reale (kW/kWp)
    y_true = inverse_scale_y(y_true_scaled, scaler)
    y_pred = inverse_scale_y(y_pred_scaled, scaler)

    # metriche reali
    mae, mse, rmse = compute_metrics(y_true, y_pred)

    # -------------------------------------------------------
    # 6. MASE in spazio reale usando il train come "insample"
    # -------------------------------------------------------
    # Qui usiamo la serie di train scalata salvata dal main
    # (data/processed/y_train_scaled.csv) e la riportiamo in spazio reale. :contentReference[oaicite:1]{index=1}
    y_train_scaled_df = pd.read_csv("data/processed/y_train_scaled.csv", index_col=0)
    y_train_scaled_arr = y_train_scaled_df.to_numpy(dtype=float)

    # inverse scaling del train (insample)
    y_train_real = inverse_scale_y(y_train_scaled_arr, scaler)

    # MASE con stagione m=24 (dati orari, naïve "stesso orario del giorno prima")
    mase_value = mase(y_true, y_pred, insample=y_train_real, m=24)

    # -------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------
    print("\n=== Test Metrics (scaled space) ===")
    print(f"MSE  (scaled): {mse_scaled:.4f}")
    print(f"RMSE (scaled): {rmse_scaled:.4f}")

    print("\n=== Test Metrics (real space, kW/kWp) ===")
    print(f"MAE  (real): {mae:.4f}")
    print(f"MSE  (real): {mse:.4f}")
    print(f"RMSE (real): {rmse:.4f}")

    print("\n=== Test Metric: MASE (real space) ===")
    print(f"MASE (m=24): {mase_value:.4f}")


if __name__ == "__main__":
    main()
