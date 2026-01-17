# evaluate.py

import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_module import PVForecastDataset, build_dataloader
from src.data_upload import load_test_datasets
from src.models import build_model
from src.config import ExperimentConfig
from src.preprocessing import (
    preprocess_pipeline,
    extract_site_coords,
    transform_ohe_with_vocab,
    apply_scaler,
)
from src.feature_engineering import (
    add_solar_features,
    add_effective_features,
    add_cloud_effect,
    add_solar_time_features,
)


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


def _plot_weekly_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    *,
    seasonality: int = 24,
    hours: int = 24 * 7,
    start_idx: int | None = None,
) -> str | None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim == 2:
        y_true_series = y_true[:, 0]
    else:
        y_true_series = y_true.ravel()

    if y_pred.ndim == 2:
        y_pred_series = y_pred[:, 0]
    else:
        y_pred_series = y_pred.ravel()

    if len(y_true_series) <= seasonality:
        return None

    if start_idx is None:
        start_idx = seasonality
    end_idx = min(start_idx + hours, len(y_true_series))
    if end_idx <= start_idx:
        return None

    true_week = y_true_series[start_idx:end_idx]
    pred_week = y_pred_series[start_idx:end_idx]
    naive_week = y_true_series[start_idx - seasonality : end_idx - seasonality]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(true_week))
    ax.plot(x, true_week, label="Ground truth", color="#1f77b4", linewidth=1.5)
    ax.plot(x, pred_week, label="Model prediction", color="#2ca02c", linewidth=1.5)
    ax.plot(x, naive_week, label=f"Naive (t-{seasonality})", color="#d62728", linewidth=1.2, linestyle="--")
    ax.set_title("Forecast vs Naive (1-step) over 7 days")
    ax.set_xlabel("Hours")
    ax.set_ylabel("kW/kWp")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def evaluate_test_sheet(cfg, device):
    """
    Valutazione sul foglio test (es. "07-12--06-13").
    Usa:
    - artifacts/{model_filename}
    - artifacts/{scaler_filename}
    - artifacts/{ohe_vocab_filename}
    - data/processed/{y_train_filename} (per MASE)
    """
    if cfg.split.mode not in {"train_val", "train_all"}:
        raise ValueError("Per test serve un modello unico: usa mode='train_val' o 'train_all'.")

    print(">>> Modalità: test sheet")

    artifacts_dir = cfg.paths.artifacts_dir
    processed_dir = cfg.paths.processed_dir
    batch_size = cfg.dataloader.batch_size

    X_raw, y_raw = load_test_datasets(
        cfg.paths.wx_path,
        cfg.paths.pv_path,
        sheet_name=cfg.test.sheet_name,
    )

    lat, lon = extract_site_coords(X_raw)
    X_base, y_base = preprocess_pipeline(
        X_raw,
        y_raw,
        fixed_offset_hours=10,
        debug=False,
        save_processed=False,
    )

    vocab_path = os.path.join(artifacts_dir, cfg.paths.ohe_vocab_filename)
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"OHE vocab non trovato: {vocab_path}")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    X_enc = transform_ohe_with_vocab(X_base, vocab)
    X_feat = add_solar_features(X_enc, lat, lon)
    X_feat = add_effective_features(X_feat)
    X_feat = add_cloud_effect(X_feat)
    X_feat = add_solar_time_features(X_feat, lat)

    scaler_path = os.path.join(artifacts_dir, cfg.paths.scaler_filename)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler non trovato: {scaler_path}")
    scaler = pickle.load(open(scaler_path, "rb"))

    X_scaled = apply_scaler(X_feat, scaler)
    y_scaled = apply_scaler(y_base, scaler, is_target=True)

    test_ds = PVForecastDataset(X_scaled, y_scaled, cfg.data)
    test_loader = build_dataloader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    sample = next(iter(test_loader))
    cfg.model.input_size = int(sample["x_hist"].shape[-1])
    cfg.model.horizon = int(test_ds.horizon)

    model = build_model(cfg.model, device)
    model_path = os.path.join(artifacts_dir, cfg.paths.model_filename)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x_hist"].to(device)
            y = batch["y_future"].cpu().numpy()
            y_hat = model(x).cpu().numpy()
            y_true_list.append(y)
            y_pred_list.append(y_hat)

    y_true_scaled = np.concatenate(y_true_list, axis=0)
    y_pred_scaled = np.concatenate(y_pred_list, axis=0)

    y_true = inverse_scale_y(y_true_scaled, scaler)
    y_pred = inverse_scale_y(y_pred_scaled, scaler)
    y_pred = np.maximum(y_pred, 0.0)

    mae, mse, rmse = compute_metrics(y_true, y_pred)

    y_train_path = os.path.join(processed_dir, cfg.paths.y_train_filename)
    if not os.path.exists(y_train_path):
        raise FileNotFoundError(f"y_train per MASE non trovato: {y_train_path}")
    y_train_scaled_df = pd.read_csv(y_train_path, index_col=0)
    y_train_scaled_arr = y_train_scaled_df.to_numpy(dtype=float)
    y_train_real = inverse_scale_y(y_train_scaled_arr, scaler)
    mase_value = mase(y_true, y_pred, insample=y_train_real, m=24)

    seasonality = 24
    hours = 24 * 7
    y_true_series = np.asarray(y_true, dtype=float)
    if y_true_series.ndim == 2:
        y_true_series = y_true_series[:, 0]
    max_start = len(y_true_series) - hours
    base_start = seasonality
    plot_starts: list[int] = []
    if max_start > base_start:
        rng = np.random.default_rng(42)
        candidate_starts = np.arange(base_start, max_start)
        if len(candidate_starts) >= 4:
            random_starts = rng.choice(candidate_starts, size=3, replace=False).tolist()
        else:
            random_starts = candidate_starts.tolist()
        plot_starts = [base_start] + random_starts
    elif max_start > 0:
        plot_starts = [base_start]

    if not plot_starts:
        print("\n[PLOT] Serie troppo corta per i grafici weekly.")
    else:
        for idx, start in enumerate(plot_starts, start=1):
            plot_path = _plot_weekly_forecast(
                y_true,
                y_pred,
                output_path=f"eda_plots/pred_vs_naive_week_{idx}.png",
                seasonality=seasonality,
                hours=hours,
                start_idx=int(start),
            )
            if plot_path is not None:
                print(f"\n[PLOT] Forecast vs naive salvato in {plot_path}")
            else:
                print("\n[PLOT] Serie troppo corta per il grafico weekly.")

    print("\n=== Test Metrics (real space, kW/kWp) ===")
    print(f"MAE  (real): {mae:.4f}")
    print(f"MSE  (real): {mse:.4f}")
    print(f"RMSE (real): {rmse:.4f}")
    print(f"MASE (m=24): {mase_value:.4f}")


__all__ = ["evaluate_test_sheet"]


def main() -> None:
    cfg = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_test_sheet(cfg, device)


if __name__ == "__main__":
    main()
