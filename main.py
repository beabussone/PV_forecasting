# main.py

import numpy as np
import os
import pickle
from src.data_upload import load_datasets
from src.EDA import run_basic_eda, analyze_feature_label_correlations
from src.preprocessing import (
    preprocess_pipeline,
    extract_site_coords,
    fit_ohe_on_train,
    transform_ohe_with_vocab,
    fit_scaler_on_train,
    apply_scaler,
)
from src.feature_engineering import (
    add_solar_features,
    add_effective_features,
    add_cloud_effect,
    add_solar_time_features,
    save_feature_engineered_X,
)

from src.models import build_model                 # deve costruire TCN da cfg.model
# modifica per tuning
from src.training import fit, evaluate_loss        # dal tuo training.py unificato
from src.hyperparameter_search import random_search, _set_by_path  # se random search abilitato

from src.data_module import (
    temporal_train_val_split,
    temporal_cv_splits_train_val,
    PVForecastDataset,
    build_dataloader,
    make_val_with_context,
)
from src.config import ExperimentConfig


def main():
    print("=== PV Forecasting Pipeline ===")

    # -----------------------------
    # Config globale dell’esperimento
    # -----------------------------
    cfg = ExperimentConfig()

    # 1) Caricamento dataset
    X_raw, y_raw = load_datasets(
        wx_path=cfg.paths.wx_path,
        pv_path=cfg.paths.pv_path,
    )
    print(f"[LOAD] X_raw: {X_raw.shape}, y_raw: {y_raw.shape}")

    # 1bis) Metadati sito (lat/lon) PRIMA che vengano droppati
    lat, lon = extract_site_coords(X_raw)
    print(f"[SITE] lat={lat}, lon={lon}")

    # 2) EDA + analisi con la label (solo stampe / info, niente modifiche)
    run_basic_eda(X_raw, y_raw)
    analyze_feature_label_correlations(X_raw, y_raw, label_col="kwp")

    # 3) Preprocessing deterministico: missing, timezone+cyc, allineamento, float32.
    #    Qui non si fa alcun fit, così i passi successivi lavorano su dati puliti ma non “sbilanciati”.
    X_base, y_base = preprocess_pipeline(
        X_raw,
        y_raw,
        fixed_offset_hours=10,
        save_processed=True,
    )
    print(f"[BASE] X_base: {X_base.shape}, y_base: {y_base.shape}")
    
    # 4) Split temporale (prima di OHE/feature engineering)
    mode = cfg.split.mode
    data_config = cfg.data
    batch_size = cfg.dataloader.batch_size
    num_workers = cfg.dataloader.num_workers
    scaling_mode = cfg.dataloader.scaling_mode

    # Blocco FE riutilizzabile per evitare duplicazione tra train/val/test o tra fold
    def fe_block(df):
        out = add_solar_features(df, lat, lon)
        out = add_effective_features(out)
        out = add_cloud_effect(out)
        out = add_solar_time_features(out, lat)
        return out

    folds_raw = []
    if mode == "train_val":
        split = temporal_train_val_split(
            X_base,
            y_base,
            train_ratio=cfg.split.train_ratio,
            val_ratio=cfg.split.val_ratio,
        )
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = split
        folds_raw.append(
            {
                "fold": 0,
                "X_train": X_train_raw,
                "X_val": X_val_raw,
                "y_train": y_train_raw,
                "y_val": y_val_raw,
            }
        )
    elif mode == "cv":
        raw_splits = temporal_cv_splits_train_val(
            X_base,
            y_base,
            n_splits=cfg.split.n_splits,
        )
        for idx, split in enumerate(raw_splits):
            X_train_raw, X_val_raw, y_train_raw, y_val_raw = split
            folds_raw.append(
                {
                    "fold": idx,
                    "X_train": X_train_raw,
                    "X_val": X_val_raw,
                    "y_train": y_train_raw,
                    "y_val": y_val_raw,
                }
            )
    else:
        raise ValueError("cfg.split.mode deve essere 'train_val' oppure 'cv'")

    # ------------------------------------------------------
    # 5–7) OHE, FE e scaling per fold
    # ------------------------------------------------------
    folds_processed = []
    for fr in folds_raw:
        # 5) OHE fittato solo sul train e applicato a val
        X_train_enc, vocab = fit_ohe_on_train(fr["X_train"])
        X_val_enc = transform_ohe_with_vocab(fr["X_val"], vocab)

        # 6) Feature engineering avanzato
        X_train_feat = fe_block(X_train_enc)
        X_val_feat = fe_block(X_val_enc)

        print(
            f"[COLUMNS][fold {fr['fold']}] FE columns ({len(X_train_feat.columns)}): "
            f"{list(X_train_feat.columns)}"
        )

        # 7) Scaling opzionale (fit SOLO su train, sia X che y)
        scaler = fit_scaler_on_train(X_train_feat, fr["y_train"], mode=scaling_mode)

        X_train_scaled = apply_scaler(X_train_feat, scaler)
        X_val_scaled = apply_scaler(X_val_feat, scaler)

        y_train_scaled = apply_scaler(fr["y_train"], scaler, is_target=True)
        y_val_scaled = apply_scaler(fr["y_val"], scaler, is_target=True)

        folds_processed.append(
            {
                "fold": fr["fold"],
                "X_train": X_train_scaled,
                "X_val": X_val_scaled,
                # modifica per tuning
                "y_train_raw": fr["y_train"],
                "y_val_raw": fr["y_val"],
                "y_train": y_train_scaled,
                "y_val": y_val_scaled,
                "scaler": scaler,
            }
        )

    # mi assicuro che esista la cartella per gli artifacts (modelli, scaler, ecc.)
    os.makedirs(cfg.paths.artifacts_dir, exist_ok=True)
    os.makedirs(cfg.paths.processed_dir, exist_ok=True)

    # ------------------------------------------------------
    # 8) Salvataggi / dataloader per train_val o CV
    # ------------------------------------------------------
    if mode == "train_val":
        p = folds_processed[0]

        # --- Salvo scaler e validation set per evaluate.py ---
        scaler_path = os.path.join(cfg.paths.artifacts_dir, cfg.paths.scaler_filename)
        with open(scaler_path, "wb") as f:
            pickle.dump(p["scaler"], f)

        np.save(
            os.path.join(cfg.paths.artifacts_dir, cfg.paths.X_val_filename),
            p["X_val"].to_numpy(dtype="float32"),
        )
        np.save(
            os.path.join(cfg.paths.artifacts_dir, cfg.paths.y_val_filename),
            p["y_val"].to_numpy(dtype="float32"),
        )

        # y scalate per MASE
        p["y_train"].to_csv(
            os.path.join(cfg.paths.processed_dir, cfg.paths.y_train_filename)
        )
        p["y_val"].to_csv(
            os.path.join(cfg.paths.processed_dir, cfg.paths.y_val_out_filename)
        )

        # Dataset + DataLoader
        train_ds = PVForecastDataset(p["X_train"], p["y_train"], data_config)

        # ✅ Validation con context dal train
        X_val_ctx, y_val_ctx, min_start = make_val_with_context(
            p["X_train"],
            p["y_train"],
            p["X_val"],
            p["y_val"],
            history=data_config.history_hours,
        )

        val_ds = PVForecastDataset(
            X_val_ctx,
            y_val_ctx,
            data_config,
            min_start_idx=min_start,   # <-- solo se hai aggiunto min_start_idx nel Dataset
        )

        train_loader = build_dataloader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        ### Modifica per tuning ###
        # Sync input_size/horizon dal primo batch (evita 32 vs 33)
        sample = next(iter(train_loader))
        cfg.model.input_size = int(sample["x_hist"].shape[-1])
        cfg.model.horizon = int(cfg.data.horizon_hours)
        ### fine modifica ###

        val_loader = build_dataloader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )


        print(
            f"[SPLIT] train: {p['X_train'].shape}, "
            f"val: {p['X_val'].shape}"
        )

        print(
            f"[DATA] train windows: {len(train_loader.dataset)} | "
            f"val: {len(val_loader.dataset)} | "
            f"hist={data_config.history_hours}h, "
            f"horizon={data_config.horizon_hours}h"
        )

        print(
            f"[LOADER] train batches: {len(train_loader)}, "
            f"val batches: {len(val_loader)}"
        )

        loaders = {"train_loader": train_loader, "val_loader": val_loader}

    else:  # mode == "cv"
        cv_loaders = []
        for p in folds_processed:
            fold_id = p["fold"]

            # Salvataggio y scalate per fold (per MASE / debug)
            y_train_path = os.path.join(
                cfg.paths.processed_dir,
                cfg.paths.y_train_fold_template.format(fold=fold_id),
            )
            y_val_path = os.path.join(
                cfg.paths.processed_dir,
                cfg.paths.y_val_out_fold_template.format(fold=fold_id),
            )

            p["y_train"].to_csv(y_train_path)
            p["y_val"].to_csv(y_val_path)

            print(f"[SAVE][fold {fold_id}] salvati y scalati in:")
            print("  ", y_train_path)
            print("  ", y_val_path)

            # Salvo scaler e validation set scalati PER FOLD per evaluate_cv
            scaler_path = os.path.join(
                cfg.paths.artifacts_dir,
                cfg.paths.scaler_fold_template.format(fold=fold_id),
            )
            with open(scaler_path, "wb") as f_sc:
                pickle.dump(p["scaler"], f_sc)

            np.save(
                os.path.join(
                    cfg.paths.artifacts_dir,
                    cfg.paths.X_val_fold_template.format(fold=fold_id),
                ),
                p["X_val"].to_numpy(dtype="float32"),
            )
            np.save(
                os.path.join(
                    cfg.paths.artifacts_dir,
                    cfg.paths.y_val_fold_template.format(fold=fold_id),
                ),
                p["y_val"].to_numpy(dtype="float32"),
            )

            # Dataset + DataLoader per fold
            train_ds = PVForecastDataset(p["X_train"], p["y_train"], data_config)

            # ✅ Validation con context dal train (per questo fold)
            X_val_ctx, y_val_ctx, min_start = make_val_with_context(
                p["X_train"],
                p["y_train"],
                p["X_val"],
                p["y_val"],
                history=data_config.history_hours,
            )

            val_ds = PVForecastDataset(
                X_val_ctx,
                y_val_ctx,
                data_config,
                min_start_idx=min_start,  # <-- solo se hai min_start_idx nel Dataset
            )

            train_loader = build_dataloader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            val_loader = build_dataloader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

            ### Modifica per tuning ###
            sample = next(iter(train_loader))
            cfg.model.input_size = int(sample["x_hist"].shape[-1])
            cfg.model.horizon = int(cfg.data.horizon_hours)

            ### fine modifica ###

            cv_loaders.append(
                {
                    "fold": fold_id,
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                }
            )

        for f, p in zip(cv_loaders, folds_processed):
            print(
                f"[CV][fold {f['fold']}] "
                f"train: {p['X_train'].shape}, val: {p['X_val'].shape}"
            )
            print(
                f"[LOADER][fold {f['fold']}] "
                f"train windows: {len(f['train_loader'].dataset)} | "
                f"val windows: {len(f['val_loader'].dataset)} | "
                f"train batches: {len(f['train_loader'])} | "
                f"val batches: {len(f['val_loader'])}"
            )

        loaders = cv_loaders

    print("=== Pipeline completata. Dataset e DataLoader pronti per il training PyTorch. ===")

    # ------------------------------------------------------
    # 9) Training vero e proprio (TCN + metriche + (opz.) random search)
    # ------------------------------------------------------
    import torch
    from torch import nn

    from src.models import build_model                 # deve costruire TCN da cfg.model
    # modifica per tuning
    from src.training import (
        fit,
        evaluate_loss,
        evaluate_metrics,
        predict_over_loader,
        compute_naive_scale_from_series,
        compute_rmse,
        compute_mase,
    )
    from src.hyperparameter_search import random_search, _set_by_path  # se random search abilitato

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # modifica per tuning
    def _inverse_scale_y(y_scaled: np.ndarray, scaler: dict) -> np.ndarray:
        if not scaler:
            return y_scaled
        mode = scaler.get("mode")
        stats = scaler.get("y_stats", {})
        if mode == "standard":
            return (y_scaled * stats.get("std", 1.0) + stats.get("mean", 0.0)).astype(np.float32)
        denom = (stats.get("max", 0.0) - stats.get("min", 0.0)) or 1e-8
        return (y_scaled * denom + stats.get("min", 0.0)).astype(np.float32)

    # modifica per tuning
    def _compute_metrics_np(y_true_np: np.ndarray, y_pred_np: np.ndarray, naive_scale: float) -> dict:
        y_true_t = torch.from_numpy(y_true_np).float()
        y_pred_t = torch.from_numpy(y_pred_np).float()
        mse = torch.mean((y_true_t - y_pred_t) ** 2).item()
        rmse = compute_rmse(y_true_t, y_pred_t)
        mase = compute_mase(y_true_t, y_pred_t, naive_scale) if naive_scale is not None else float("nan")
        return {"loss": mse, "rmse": rmse, "mase": mase}

    # modifica per tuning
    def _maybe_naive_scale(series, m: int):
        if series is None:
            return None
        return compute_naive_scale_from_series(series, m=m)

    # helper: set input_size/horizon DAL BATCH (evita 32 vs 33 quando include_past_target=True)
    def _sync_dims_from_loader(train_loader):
        sample = next(iter(train_loader))
        cfg.model.input_size = int(sample["x_hist"].shape[-1])
        cfg.model.horizon = int(cfg.data.horizon_hours)

    # helper: esegue (opzionale) random search + fit finale
    def _train_with_optional_random_search(train_loader, val_loader):
        # 1) sync dims (sempre)
        _sync_dims_from_loader(train_loader)

        best_params = None

        # 2) random search (se presente in cfg)
        rs_cfg = getattr(cfg, "random_search", None)
        if rs_cfg is not None and getattr(rs_cfg, "enabled", False):
            rs = random_search(
                model_builder=lambda model_cfg: build_model(model_cfg, device=None),
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg,
                search_space=getattr(rs_cfg, "search_space", None),
                n_trials=int(getattr(rs_cfg, "n_trials", 20)),
                metric=str(getattr(rs_cfg, "metric", "loss")),
                mode=str(getattr(rs_cfg, "mode", "min")),
                seed=int(getattr(rs_cfg, "seed", 42)),
                device=device,
                verbose=bool(getattr(rs_cfg, "verbose", True)),
            )
            best_params = rs.best_params

            # applico i best params al cfg "vero"
            for k, v in best_params.items():
                _set_by_path(cfg, k, v)

            # re-sync dims (in caso random search abbia toccato qualcosa)
            _sync_dims_from_loader(train_loader)

        # 3) build model finale + fit finale
        model = build_model(cfg.model, device=None)

        # y insample per MASE (nello stesso spazio della loss: scaled)
        y_train_insample = getattr(getattr(train_loader, "dataset", None), "y_values", None)

        fit_result = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=int(cfg.training.epochs),
            lr=float(cfg.training.lr),
            device=device,
            loss_fn=nn.MSELoss(),
            y_train_insample=y_train_insample,
            mase_m=int(getattr(cfg.baseline, "mase_seasonal_m", 24)),
            keep_best_on_val=True,
            # modifica per tuning
            early_stopping=bool(getattr(cfg.training, "early_stopping", False)),
            patience=int(getattr(cfg.training, "patience", 5)),
            min_delta=float(getattr(cfg.training, "min_delta", 0.0)),
            verbose=True,
        )

        return fit_result, best_params


    if mode == "train_val":
        # ---- Train/Val ----
        train_loader = loaders["train_loader"]
        val_loader = loaders["val_loader"]

        fit_result, best_params = _train_with_optional_random_search(train_loader, val_loader)

        # modifica per tuning
        p = folds_processed[0]
        y_train_insample = getattr(getattr(train_loader, "dataset", None), "y_values", None)
        naive_scale_scaled = _maybe_naive_scale(
            y_train_insample, int(getattr(cfg.baseline, "mase_seasonal_m", 24))
        )
        metrics_scaled = evaluate_metrics(
            fit_result.model, val_loader, device, nn.MSELoss(), naive_scale=naive_scale_scaled
        )

        y_true_scaled, y_pred_scaled = predict_over_loader(fit_result.model, val_loader, device)
        y_true_unscaled = _inverse_scale_y(y_true_scaled, p["scaler"])
        y_pred_unscaled = _inverse_scale_y(y_pred_scaled, p["scaler"])
        y_train_raw = p["y_train_raw"].to_numpy(dtype=np.float32)
        naive_scale_unscaled = _maybe_naive_scale(
            y_train_raw, int(getattr(cfg.baseline, "mase_seasonal_m", 24))
        )
        metrics_unscaled = _compute_metrics_np(y_true_unscaled, y_pred_unscaled, naive_scale_unscaled)

        print(
            f"[METRICS] Val MSE (scaled): {metrics_scaled['loss']:.4f} | "
            f"Val MASE (scaled): {metrics_scaled['mase']:.4f}"
        )
        print(
            f"[METRICS] Val MSE (unscaled): {metrics_unscaled['loss']:.4f} | "
            f"Val MASE (unscaled): {metrics_unscaled['mase']:.4f}"
        )

        # salva best model
        model_path = os.path.join(cfg.paths.artifacts_dir, cfg.paths.model_filename)
        torch.save(fit_result.model.state_dict(), model_path)
        print(f"[SAVE] modello salvato in {model_path}")

        if best_params:
            print("[RANDOM_SEARCH] best_params:", best_params)

    else:
        # ---- CV ----
        val_scores = []
        for fold_data, p in zip(loaders, folds_processed):
            fold_id = fold_data["fold"]
            print(f"=== Training fold {fold_id} ===")

            fit_result, best_params = _train_with_optional_random_search(
                fold_data["train_loader"],
                fold_data["val_loader"],
            )

            # modifica per tuning
            y_train_insample = getattr(getattr(fold_data["train_loader"], "dataset", None), "y_values", None)
            naive_scale_scaled = _maybe_naive_scale(
                y_train_insample, int(getattr(cfg.baseline, "mase_seasonal_m", 24))
            )
            metrics_scaled = evaluate_metrics(
                fit_result.model,
                fold_data["val_loader"],
                device,
                nn.MSELoss(),
                naive_scale=naive_scale_scaled,
            )

            y_true_scaled, y_pred_scaled = predict_over_loader(
                fit_result.model, fold_data["val_loader"], device
            )
            y_true_unscaled = _inverse_scale_y(y_true_scaled, p["scaler"])
            y_pred_unscaled = _inverse_scale_y(y_pred_scaled, p["scaler"])
            y_train_raw = p["y_train_raw"].to_numpy(dtype=np.float32)
            naive_scale_unscaled = _maybe_naive_scale(
                y_train_raw, int(getattr(cfg.baseline, "mase_seasonal_m", 24))
            )
            metrics_unscaled = _compute_metrics_np(y_true_unscaled, y_pred_unscaled, naive_scale_unscaled)

            val_scores.append(metrics_scaled["loss"])
            print(
                f"[METRICS][fold {fold_id}] Val MSE (scaled): {metrics_scaled['loss']:.4f} | "
                f"Val MASE (scaled): {metrics_scaled['mase']:.4f}"
            )
            print(
                f"[METRICS][fold {fold_id}] Val MSE (unscaled): {metrics_unscaled['loss']:.4f} | "
                f"Val MASE (unscaled): {metrics_unscaled['mase']:.4f}"
            )

            model_path = os.path.join(
                cfg.paths.artifacts_dir,
                cfg.paths.model_fold_template.format(fold=fold_id),
            )
            torch.save(fit_result.model.state_dict(), model_path)
            print(f"[SAVE][fold {fold_id}] modello salvato in {model_path}")

            if best_params:
                print(f"[RANDOM_SEARCH][fold {fold_id}] best_params:", best_params)

        if val_scores:
            val_mean = float(np.mean(val_scores))
            val_std = float(np.std(val_scores))
            print(f"[CV][VAL] mean MSE: {val_mean:.4f} | std: {val_std:.4f}")


if __name__ == "__main__":
    main()
