# evaluate.py

import os
import pickle
import numpy as np
import pandas as pd
import torch

from src.data_module import PVForecastDataset, build_dataloader
from src.models import build_model
from src.config import ExperimentConfig
from src.training import evaluate_loss as eval_loss


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


def evaluate_validation_split(cfg, device):
    """
    Valutazione sul validation set (split train/val 80/20).
    Usa:
    - artifacts/{scaler_filename}
    - artifacts/{X_val_filename}
    - artifacts/{y_val_filename}
    - data/processed/{y_train_filename} (per MASE)
    - artifacts/{model_filename}
    """
    print(">>> Modalità: train_val")

    artifacts_dir = cfg.paths.artifacts_dir
    processed_dir = cfg.paths.processed_dir
    batch_size = cfg.dataloader.batch_size

    # 1. Carico scaler salvato
    scaler_path = os.path.join(artifacts_dir, cfg.paths.scaler_filename)
    scaler = pickle.load(open(scaler_path, "rb"))

    # 2. Carico il validation set salvato dal main (già FE + scaling)
    X_val_arr = np.load(os.path.join(artifacts_dir, cfg.paths.X_val_filename))
    y_val_arr = np.load(os.path.join(artifacts_dir, cfg.paths.y_val_filename))

    # Converto in DataFrame per soddisfare PVForecastDataset
    X_val = pd.DataFrame(X_val_arr)

    # y_val probabilmente è shape (N, 1) – ci assicuriamo che sia 2D
    if y_val_arr.ndim == 1:
        y_val_arr = y_val_arr.reshape(-1, 1)
    y_val = pd.DataFrame(y_val_arr, columns=["kwp"])  # nome colonna coerente

    val_ds = PVForecastDataset(
        X_val,
        y_val,
        cfg.data,
    )

    # LOADER
    val_loader = build_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # 3. Ricostruisco modello e carico pesi
    cfg.model.input_size = val_ds.X_values.shape[1]
    cfg.model.horizon = val_ds.horizon

    model = build_model(cfg.model, device)
    model_path = os.path.join(artifacts_dir, cfg.paths.model_filename)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Metriche nello spazio scalato
    criterion = torch.nn.MSELoss()
    mse_scaled = eval_loss(model, val_loader, criterion, device)
    rmse_scaled = float(np.sqrt(mse_scaled))

    # 5. Predizioni + inverse scaling (validation)
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for batch in val_loader:
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

    # 6. MASE in spazio reale usando il train come "insample"
    y_train_path = os.path.join(processed_dir, cfg.paths.y_train_filename)
    y_train_scaled_df = pd.read_csv(y_train_path, index_col=0)
    y_train_scaled_arr = y_train_scaled_df.to_numpy(dtype=float)

    # inverse scaling del train (insample)
    y_train_real = inverse_scale_y(y_train_scaled_arr, scaler)

    mase_value = mase(y_true, y_pred, insample=y_train_real, m=24)

    # OUTPUT
    print("\n=== Validation Metrics (scaled space) ===")
    print(f"MSE  (scaled): {mse_scaled:.4f}")
    print(f"RMSE (scaled): {rmse_scaled:.4f}")

    print("\n=== Validation Metrics (real space, kW/kWp) ===")
    print(f"MAE  (real): {mae:.4f}")
    print(f"MSE  (real): {mse:.4f}")
    print(f"RMSE (real): {rmse:.4f}")

    print("\n=== Validation Metric: MASE (real space) ===")
    print(f"MASE (m=24): {mase_value:.4f}")


def evaluate_cv(cfg, device):
    """
    Valutazione per la cross-validation temporale (solo train/val).
    Per ogni fold k usa:
    - artifacts/{scaler_fold_template.format(fold=k)}
    - artifacts/{X_val_fold_template.format(fold=k)}
    - artifacts/{y_val_fold_template.format(fold=k)}
    - data/processed/{y_train_fold_template.format(fold=k)}
    - artifacts/{model_fold_template.format(fold=k)}

    Stampa metriche per fold + media complessiva.
    """
    print(">>> Modalità: cross-validation (cv)")

    n_splits = cfg.split.n_splits
    artifacts_dir = cfg.paths.artifacts_dir
    processed_dir = cfg.paths.processed_dir
    batch_size = cfg.dataloader.batch_size

    all_mse_scaled = []
    all_rmse_scaled = []
    all_mae = []
    all_mse = []
    all_rmse = []
    all_mase = []

    for fold_id in range(n_splits):
        print(f"\n--- Fold {fold_id} ---")

        scaler_path = os.path.join(
            artifacts_dir, cfg.paths.scaler_fold_template.format(fold=fold_id)
        )
        X_val_path = os.path.join(
            artifacts_dir, cfg.paths.X_val_fold_template.format(fold=fold_id)
        )
        y_val_path = os.path.join(
            artifacts_dir, cfg.paths.y_val_fold_template.format(fold=fold_id)
        )
        model_path = os.path.join(
            artifacts_dir, cfg.paths.model_fold_template.format(fold=fold_id)
        )
        y_train_path = os.path.join(
            processed_dir, cfg.paths.y_train_fold_template.format(fold=fold_id)
        )

        # se mancano i file di un fold, lo skippiamo (non crasha tutto)
        if not (
            os.path.exists(scaler_path)
            and os.path.exists(X_val_path)
            and os.path.exists(y_val_path)
            and os.path.exists(model_path)
            and os.path.exists(y_train_path)
        ):
            print(f"[WARN] Artifacts mancanti per fold {fold_id}, skip.")
            continue

        scaler = pickle.load(open(scaler_path, "rb"))
        X_val_arr = np.load(X_val_path)
        y_val_arr = np.load(y_val_path)

        X_val = pd.DataFrame(X_val_arr)
        if y_val_arr.ndim == 1:
            y_val_arr = y_val_arr.reshape(-1, 1)
        y_val = pd.DataFrame(y_val_arr, columns=["kwp"])

        val_ds = PVForecastDataset(X_val, y_val, cfg.data)
        val_loader = build_dataloader(val_ds, batch_size=batch_size, shuffle=False)

        cfg.model.input_size = val_ds.X_values.shape[1]
        cfg.model.horizon = val_ds.horizon

        model = build_model(cfg.model, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        criterion = torch.nn.MSELoss()
        mse_scaled = eval_loss(model, val_loader, criterion, device)
        rmse_scaled = float(np.sqrt(mse_scaled))

        y_true_list = []
        y_pred_list = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x_hist"].to(device)
                y = batch["y_future"].cpu().numpy()
                y_hat = model(x).cpu().numpy()
                y_true_list.append(y)
                y_pred_list.append(y_hat)

        y_true_scaled = np.concatenate(y_true_list, axis=0)
        y_pred_scaled = np.concatenate(y_pred_list, axis=0)

        y_true = inverse_scale_y(y_true_scaled, scaler)
        y_pred = inverse_scale_y(y_pred_scaled, scaler)

        mae, mse_v, rmse_v = compute_metrics(y_true, y_pred)

        # MASE per il fold
        y_train_scaled_df = pd.read_csv(y_train_path, index_col=0)
        y_train_scaled_arr = y_train_scaled_df.to_numpy(dtype=float)
        y_train_real = inverse_scale_y(y_train_scaled_arr, scaler)
        mase_value = mase(y_true, y_pred, insample=y_train_real, m=24)

        print("Fold metrics (scaled):")
        print(f"  MSE  (scaled): {mse_scaled:.4f}")
        print(f"  RMSE (scaled): {rmse_scaled:.4f}")
        print("Fold metrics (real space, kW/kWp):")
        print(f"  MAE  (real): {mae:.4f}")
        print(f"  MSE  (real): {mse_v:.4f}")
        print(f"  RMSE (real): {rmse_v:.4f}")
        print(f"  MASE (m=24): {mase_value:.4f}")

        all_mse_scaled.append(mse_scaled)
        all_rmse_scaled.append(rmse_scaled)
        all_mae.append(mae)
        all_mse.append(mse_v)
        all_rmse.append(rmse_v)
        all_mase.append(mase_value)

    if not all_mse_scaled:
        print("\n[ERROR] Nessun fold valido valutato (artifacts mancanti?).")
        return

    print("\n=== CV – Average metrics over folds ===")
    print(f"MSE  (scaled) mean: {np.mean(all_mse_scaled):.4f}")
    print(f"RMSE (scaled) mean: {np.mean(all_rmse_scaled):.4f}")
    print(f"MAE  (real)   mean: {np.mean(all_mae):.4f}")
    print(f"MSE  (real)   mean: {np.mean(all_mse):.4f}")
    print(f"RMSE (real)   mean: {np.mean(all_rmse):.4f}")
    print(f"MASE (m=24)   mean: {np.mean(all_mase):.4f}")


def main():
    print("=== Evaluate Validation Set ===")

    cfg = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.split.mode == "train_val":
        evaluate_validation_split(cfg, device)
    elif cfg.split.mode == "cv":
        evaluate_cv(cfg, device)
    else:
        raise ValueError("cfg.split.mode deve essere 'train_val' oppure 'cv'")


if __name__ == "__main__":
    main()
