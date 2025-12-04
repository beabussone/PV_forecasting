# main.py

import torch
import numpy as np
import os
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
    temporal_train_val_test_split,
    temporal_cv_splits,
    PVForecastDataset,
    build_dataloader,
)


from src.config import ExperimentConfig
from src.models import build_model
from src.training import fit
import pickle

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
    '''raw_splits = prepare_data_splits(
        X_base,
        y_base,
        mode=mode,
        train_ratio=cfg.split.train_ratio,
        val_ratio=cfg.split.val_ratio,
        n_splits=cfg.split.n_splits,
    )'''

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

    # Normalizza gli split in una lista di fold con test (TVT = 1 fold, CV = n fold)
    # Normalizza gli split in una lista di fold con test (TVT = 1 fold, CV = n fold)
    folds_raw = []
    if mode == "train_val_test":
        raw_splits = temporal_train_val_test_split(
            X_base,
            y_base,
            train_ratio=cfg.split.train_ratio,
            val_ratio=cfg.split.val_ratio,
        )
        X_train, X_val, X_test, y_train, y_val, y_test = raw_splits
        folds_raw.append(
            {
                "fold": 0,
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
            }
        )
    elif mode == "cv":
        raw_splits = temporal_cv_splits(
            X_base,
            y_base,
            n_splits=cfg.split.n_splits,
        )
        for idx, split in enumerate(raw_splits):
            X_train, X_val, X_test, y_train, y_val, y_test = split
            folds_raw.append(
                {
                    "fold": idx,
                    "X_train": X_train,
                    "X_val": X_val,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_val": y_val,
                    "y_test": y_test,
                }
            )
    else:
        raise ValueError("mode deve essere 'train_val_test' oppure 'cv'")
    
    # Applica OHE/FE/scaling per ogni fold in modo indipendente (fit solo sul train del fold)
    folds_processed = []
    for fr in folds_raw:
        # 5) OHE fittato solo sul train e applicato a val/test
        X_train_enc, vocab = fit_ohe_on_train(fr["X_train"])
        X_val_enc = transform_ohe_with_vocab(fr["X_val"], vocab)
        X_test_enc = transform_ohe_with_vocab(fr["X_test"], vocab) if fr["X_test"] is not None else None

        # 6) Feature engineering avanzato
        X_train_feat = fe_block(X_train_enc)
        X_val_feat = fe_block(X_val_enc)
        X_test_feat = fe_block(X_test_enc) if X_test_enc is not None else None

        print(
            f"[COLUMNS][fold {fr['fold']}] FE columns ({len(X_train_feat.columns)}): "
            f"{list(X_train_feat.columns)}"
        )

        # 7) Scaling opzionale (fit SOLO su train, sia X che y)
        scaler = fit_scaler_on_train(X_train_feat, fr["y_train"], mode=scaling_mode)

        X_train_scaled = apply_scaler(X_train_feat, scaler)
        X_val_scaled   = apply_scaler(X_val_feat, scaler)
        X_test_scaled  = apply_scaler(X_test_feat, scaler) if X_test_feat is not None else None

        y_train_scaled = apply_scaler(fr["y_train"], scaler, is_target=True)
        y_val_scaled   = apply_scaler(fr["y_val"], scaler, is_target=True)
        y_test_scaled  = apply_scaler(fr["y_test"], scaler, is_target=True) if fr["y_test"] is not None else None

        folds_processed.append(
            {
                "fold": fr["fold"],
                "X_train": X_train_scaled,
                "X_val": X_val_scaled,
                "X_test": X_test_scaled,
                "y_train": y_train_scaled,
                "y_val": y_val_scaled,
                "y_test": y_test_scaled,
                "scaler": scaler,
            }
        )
        

    if mode == "train_val_test":
        p = folds_processed[0]
        
         # --- Salvo scaler e test set per evaluate.py ---
        os.makedirs("artifacts", exist_ok=True)

        # 1) scaler (serve per inverse-transform di y)
        with open("artifacts/scaler.pkl", "wb") as f:
            pickle.dump(p["scaler"], f)

        # 2) X_test e y_test già scalati (come usati dal modello) --> DEVO FARLO ANCHE IN CROSS VALIDATION
        np.save("artifacts/X_test_scaled.npy", p["X_test"].to_numpy(dtype="float32"))
        np.save("artifacts/y_test_scaled.npy", p["y_test"].to_numpy(dtype="float32"))
        
        # 8) Salvataggi facoltativi dei dataset con feature ingegnerizzate
        p["y_train"].to_csv("data/processed/y_train_scaled.csv")
        p["y_val"].to_csv("data/processed/y_val_scaled.csv")
        p["y_test"].to_csv("data/processed/y_test_scaled.csv")

        save_feature_engineered_X(p["X_train"], out_path=cfg.paths.X_train_feat_out)
        save_feature_engineered_X(p["X_val"], out_path=cfg.paths.X_val_feat_out)
        save_feature_engineered_X(p["X_test"], out_path=cfg.paths.X_test_feat_out)

        train_ds = PVForecastDataset(p["X_train"], p["y_train"], data_config)
        val_ds = PVForecastDataset(p["X_val"], p["y_val"], data_config)
        test_ds = PVForecastDataset(p["X_test"], p["y_test"], data_config)

        train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = build_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = build_dataloader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(
            f"[SPLIT] train: {p['X_train'].shape}, "
            f"val: {p['X_val'].shape}, "
            f"test: {p['X_test'].shape}"
        )

        print(
            f"[DATA] train windows: {len(train_loader.dataset)} | "
            f"val: {len(val_loader.dataset)} | "
            f"test: {len(test_loader.dataset)} | "
            f"hist={data_config.history_hours}h, "
            f"horizon={data_config.horizon_hours}h"
        )

        print(
            f"[LOADER] train batches: {len(train_loader)}, "
            f"val batches: {len(val_loader)}, "
            f"test batches: {len(test_loader)}"
        )

    else:
        # 8) DataLoader per CV (include test comune)
        cv_loaders = []
        for p in folds_processed:
            # salvataggio y
            fold_id = p["fold"]

            # Percorsi dei file
            y_train_path = f"data/processed/y_train_scaled_fold{fold_id}.csv"
            y_val_path   = f"data/processed/y_val_scaled_fold{fold_id}.csv"
            y_test_path  = f"data/processed/y_test_scaled_fold{fold_id}.csv"

            # Salvataggio
            p["y_train"].to_csv(y_train_path)
            p["y_val"].to_csv(y_val_path)
            if p["y_test"] is not None:
                p["y_test"].to_csv(y_test_path)

            print(f"[SAVE][fold {fold_id}] salvati y scalati in:")
            print("  ", y_train_path)
            print("  ", y_val_path)
            if p["y_test"] is not None:
                print("  ", y_test_path)

            
            train_ds = PVForecastDataset(p["X_train"], p["y_train"], data_config)
            val_ds = PVForecastDataset(p["X_val"], p["y_val"], data_config)
            test_ds = PVForecastDataset(p["X_test"], p["y_test"], data_config)

            train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = build_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_loader = build_dataloader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            cv_loaders.append(
                {
                    "fold": p["fold"],
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "test_loader": test_loader,
                }
            )

        for f, p in zip(cv_loaders, folds_processed):
            print(
                f"[CV][fold {f['fold']}] "
                f"train: {p['X_train'].shape}, val: {p['X_val'].shape}, test: {p['X_test'].shape}"
            )
            print(
                f"[LOADER][fold {f['fold']}] "
                f"train windows: {len(f['train_loader'].dataset)} | "
                f"val windows: {len(f['val_loader'].dataset)} | "
                f"test windows: {len(f['test_loader'].dataset)} | "
                f"train batches: {len(f['train_loader'])} | "
                f"val batches: {len(f['val_loader'])} | "
                f"test batches: {len(f['test_loader'])}"
            )
    print("=== Pipeline completata. Dataset e DataLoader pronti per il training PyTorch. ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if mode == "train_val_test":
        # 1) recupero info dal dataset
        train_dataset = train_loader.dataset
        cfg.model.input_size = train_dataset.X_values.shape[1]
        cfg.model.horizon = train_dataset.horizon

        # 2) costruzione modello a partire dalla config
        model = build_model(cfg.model, device=device)

        # 3) training
        result = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg.training,
            device=device,
        )

        # salvo il modello migliore
        best_model = result["model"]
        torch.save(best_model.state_dict(), "artifacts/model_gru.pth")


    elif mode == "cv":
        # Cross-validation: ciclo sui fold
        for fold_data in cv_loaders:
            print(f"=== Training fold {fold_data['fold']} ===")

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

            

if __name__ == "__main__":
    main()
