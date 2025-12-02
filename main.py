# main.py

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
    PVDataConfig,
    PVForecastDataset,
    build_dataloader,
    temporal_train_val_test_split,
)

def main():
    print("=== PV Forecasting Pipeline ===")

    # 1) Caricamento dataset
    X_raw, y_raw = load_datasets(
        wx_path="data/wx_dataset.xlsx",
        pv_path="data/pv_dataset.xlsx"
    )
    print(f"[LOAD] X_raw: {X_raw.shape}, y_raw: {y_raw.shape}")

    # 1bis) Metadati sito (lat/lon) PRIMA che vengano droppati
    lat, lon = extract_site_coords(X_raw)
    print(f"[SITE] lat={lat}, lon={lon}")

    # 2) EDA + analisi con la label (solo stampe / info, niente modifiche)
    run_basic_eda(X_raw, y_raw)
    analyze_feature_label_correlations(X_raw, y_raw, label_col="kwp")

    # 3) Preprocessing deterministico: missing, timezone+cyc, allineamento, float32
    X_base, y_base = preprocess_pipeline(X_raw, y_raw, fixed_offset_hours=10, save_processed=True)
    print(f"[BASE] X_base: {X_base.shape}, y_base: {y_base.shape}")

    # 4) Split temporale
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(
        X_base, y_base, train_ratio=0.7, val_ratio=0.15
    )
    print(f"[SPLIT] train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # 5) OHE fittato solo sul train e applicato a val/test
    X_train_enc, vocab = fit_ohe_on_train(X_train)
    X_val_enc = transform_ohe_with_vocab(X_val, vocab)
    X_test_enc = transform_ohe_with_vocab(X_test, vocab)

    # 6) Feature engineering avanzato su ciascuno split
    def fe_block(df):
        out = add_solar_features(df, lat, lon)
        out = add_effective_features(out)
        out = add_cloud_effect(out)
        out = add_solar_time_features(out, lat)
        return out

    X_train_feat = fe_block(X_train_enc)
    X_val_feat = fe_block(X_val_enc)
    X_test_feat = fe_block(X_test_enc)

    # 7) Scaling opzionale (standard/minmax/None) solo sui float
    scaling_mode = "standard"
    scaler = fit_scaler_on_train(X_train_feat, mode=scaling_mode)
    X_train_scaled = apply_scaler(X_train_feat, scaler)
    X_val_scaled = apply_scaler(X_val_feat, scaler)
    X_test_scaled = apply_scaler(X_test_feat, scaler)

    # 8) Salvataggi facoltativi
    save_feature_engineered_X(X_train_scaled, out_path="data/processed/X_train_feat.csv")
    save_feature_engineered_X(X_val_scaled, out_path="data/processed/X_val_feat.csv")
    save_feature_engineered_X(X_test_scaled, out_path="data/processed/X_test_feat.csv")

    # 9) Dataset/DataLoader PyTorch
    data_config = PVDataConfig(
        history_hours=72,
        horizon_hours=24,
        include_future_covariates=False,
    )

    train_ds = PVForecastDataset(X_train_scaled, y_train, data_config)
    val_ds = PVForecastDataset(X_val_scaled, y_val, data_config)
    test_ds = PVForecastDataset(X_test_scaled, y_test, data_config)

    train_loader = build_dataloader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = build_dataloader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = build_dataloader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    print(
        f"[DATA] train windows: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)} "
        f"| hist={data_config.history_hours}h, horizon={data_config.horizon_hours}h"
    )
    print(f"[DATA] train batches: {len(train_loader)}, val batches: {len(val_loader)}, test batches: {len(test_loader)}")

    print("=== Pipeline completata. Dataset e DataLoader pronti per il training PyTorch. ===")


if __name__ == "__main__":
    main()
