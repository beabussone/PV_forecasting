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
from src.data_module import (
    temporal_train_val_split,
    temporal_cv_splits_train_val,
    PVForecastDataset,
    build_dataloader,
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
        val_ds = PVForecastDataset(p["X_val"], p["y_val"], data_config)

        train_loader = build_dataloader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
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
            val_ds = PVForecastDataset(p["X_val"], p["y_val"], data_config)

            train_loader = build_dataloader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            val_loader = build_dataloader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

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

    '''# ------------------------------------------------------
    # 9) Training vero e proprio
    # ------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "train_val":
        # recupero info dal dataset
        train_dataset = loaders["train_loader"].dataset
        cfg.model.input_size = train_dataset.X_values.shape[1]
        cfg.model.horizon = train_dataset.horizon

        model = build_model(cfg.model, device=device)

        result = fit(
            model=model,
            train_loader=loaders["train_loader"],
            val_loader=loaders["val_loader"],
            config=cfg.training,
            device=device,
        )

        best_model = result["model"]
        criterion = nn.MSELoss()
        val_loss = evaluate(best_model, fold_data["val_loader"], criterion, device)
        val_scores.append(val_loss)
        print(f"[METRICS][fold {fold_id}] Val MSE: {val_loss:.4f}")
        
    else:  # cv
        val_scores = []
        for p, fold_data in zip(folds_processed, loaders):
            fold_id = fold_data["fold"]
            print(f"=== Training fold {fold_id} ===")

            train_dataset = fold_data["train_loader"].dataset
            cfg.model.input_size = train_dataset.X_values.shape[1]
            cfg.model.horizon = train_dataset.horizon

            model = build_model(cfg.model, device=device)

            result = fit(
                model=model,
                train_loader=fold_data["train_loader"],
                val_loader=fold_data["val_loader"],
                config=cfg.training,
                device=device,
            )

            best_model = result["model"]
            criterion = nn.MSELoss()
            val_loss = evaluate(best_model, fold_data["val_loader"], criterion, device)
            val_scores.append(val_loss)
            print(f"[METRICS][fold {fold_id}] Val MSE: {val_loss:.4f}")

            model_path = os.path.join(
                cfg.paths.artifacts_dir,
                cfg.paths.model_fold_template.format(fold=fold_id),
            )
            torch.save(best_model.state_dict(), model_path)
            print(f"[SAVE][fold {fold_id}] modello salvato in {model_path}")

        if val_scores:
            val_mean = float(np.mean(val_scores))
            val_std = float(np.std(val_scores))
            print(f"[CV][VAL] mean MSE: {val_mean:.4f} | std: {val_std:.4f}")'''

if __name__ == "__main__":
    main()
