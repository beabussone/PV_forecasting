# main.py

from src.data_upload import load_datasets
from src.EDA import run_basic_eda, analyze_feature_label_correlations
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import add_solar_features, add_effective_features
from src.preprocessing import extract_site_coords
from src.feature_engineering import save_feature_engineered_X

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

    # 3) Preprocessing completo: missing, OHE+other, timezone+cyc, allineamento, float32
    X_proc, y_proc = preprocess_pipeline(X_raw, y_raw, fixed_offset_hours=10, save_processed=True)
    print(f"[FINAL] X_proc: {X_proc.shape}, y_proc: {y_proc.shape}")
    
    # 4) Feature engineering avanzato su X già pulito
    X_feat = add_solar_features(X_proc, lat, lon)
    X_feat = add_effective_features(X_feat)
    print(f"[FINAL] X_feat: {X_feat.shape}")
    print(f"[FINAL] X_feat columns: {X_feat.columns}")

    # salvataggi di X_feat
    save_feature_engineered_X(X_feat)
    
    print("=== Pipeline completata ===")


if __name__ == "__main__":
    main()