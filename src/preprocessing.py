# src/preprocess.py

import numpy as np
import pandas as pd
from datetime import timezone, timedelta
from pathlib import Path


# --- 1. Funzioni semplici: missing + OHE+other --- #

DEFAULT_RARE_WEATHER = [
    "haze", "light intensity shower rain", "heavy intensity rain",
    "shower rain", "light intensity drizzle", "mist", "smoke",
    "fog", "thunderstorm", "thunderstorm with rain",
    "drizzle", "proximity squalls"
]


def fill_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """Esempio: rain_1h → 0."""
    X = X.copy()
    if "rain_1h" in X.columns:
        X["rain_1h"] = X["rain_1h"].fillna(0)
    return X

# --- helper per ottenere i metadati del sito (lat/lon) --- #

def extract_site_coords(X_raw: pd.DataFrame) -> tuple[float, float]:
    """
    Estrae (lat, lon) da X_raw assumendo che siano costanti
    su tutto il dataset.
    """
    if "lat" not in X_raw.columns or "lon" not in X_raw.columns:
        raise ValueError("Impossibile estrarre lat/lon: colonne mancanti in X_raw.")

    lat = float(X_raw["lat"].iloc[0])
    lon = float(X_raw["lon"].iloc[0])
    return lat, lon

def ohe_weather_description(
    X: pd.DataFrame,
    rare_list=DEFAULT_RARE_WEATHER
) -> pd.DataFrame:
    """
    Aggrega alcune classi rare di weather_description in 'other'
    e applica One-Hot Encoding.
    """
    X = X.copy()

    if "weather_description" in X.columns:
        X["weather_description"] = X["weather_description"].replace(rare_list, "other")
        X = pd.get_dummies(
            X,
            columns=["weather_description"],
            prefix="weather",
            dummy_na=False,
            dtype=int,
        )

    # --- helper per ottenere i metadati del sito (lat/lon) --- #
    
    # Drop lat/lon se presenti
    X = X.drop(columns=[c for c in ["lat", "lon"] if c in X.columns], errors="ignore")

    return X


# --- 2. Funzione principale: timezone + cyclical encoding + align --- #

def process_all_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    x_time_col: str = "dt_iso",
    y_time_col: str | None = "datetime",
    fixed_offset_hours: int = 10,
    debug: bool = True,
):
    """
    - Porta X e y a un fuso fisso UTC+10 (niente DST).
    - Arrotonda tutto all'ora esatta (round('H')) per evitare 02:59:59.985 ecc.
    - Crea encoding ciclico (ora e mese) su X.
    - Imposta il tempo come indice e allinea X e y.
    - NON fa encoding ciclico sulla y.
    """

    fixed_tz = timezone(timedelta(hours=fixed_offset_hours))
    X = X.copy()
    y = y.copy()

    if debug:
        print("\n=== PROCESSING TEMPORALE + CICLICO ===")

    # --- X: gestione dt_iso --- #
    if x_time_col not in X.columns:
        raise ValueError(f"Colonna '{x_time_col}' non trovata in X.")

    if debug:
        print(f"[X] Esempio dt_iso PRIMA: {X[x_time_col].iloc[0]}")

    # dt_iso è in UTC → convertiamo a +10 e arrotondiamo all'ora
    X[x_time_col] = (
        pd.to_datetime(X[x_time_col], utc=True, errors="coerce")
          .dt.tz_convert(fixed_tz)
          .dt.round("h")
    )
    X = X.dropna(subset=[x_time_col])

    if debug:
        print(f"[X] Esempio dt_iso DOPO : {X[x_time_col].iloc[0]}")

    # encoding ciclico su ora e mese
    hours = X[x_time_col].dt.hour.astype(int)
    months = X[x_time_col].dt.month.astype(int)

    X["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    X["month_sin"] = np.sin(2 * np.pi * months / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * months / 12.0)

    # metto dt_iso come indice
    X = (X.set_index(x_time_col)
           .sort_index()
           .loc[~X.index.duplicated(keep="first")])

    # --- y: già in +10 ma senza timezone → localize + round --- #
    if y_time_col is None:
        y_time_col = "datetime" if "datetime" in y.columns else y.columns[0]

    if debug:
        print(f"[y] colonna tempo usata: {y_time_col}")
        print(f"[y] Esempio datetime PRIMA: {y[y_time_col].iloc[0]}")

    y[y_time_col] = pd.to_datetime(y[y_time_col], errors="coerce")

    # Label già in +10 ma naive → localizziamo direttamente a fixed_tz
    if getattr(y[y_time_col].dt, "tz", None) is None:
        y[y_time_col] = y[y_time_col].dt.tz_localize(fixed_tz)
    else:
        y[y_time_col] = y[y_time_col].dt.tz_convert(fixed_tz)

    y[y_time_col] = y[y_time_col].dt.round("h")

    y = (y.dropna(subset=[y_time_col])
           .set_index(y_time_col)
           .sort_index()
           .loc[~y.index.duplicated(keep="first")])

    if debug:
        print(f"[y] Esempio datetime DOPO : {y.index[0]}")

    # --- allineamento --- #
    X_aligned, y_aligned = X.align(y, join="inner", axis=0)

    if debug:
        only_x = len(X.index.difference(y.index))
        only_y = len(y.index.difference(X.index))
        print(f"[MATCH] comuni: {len(X_aligned)} | solo X: {only_x} | solo y: {only_y}")

    return X_aligned, y_aligned


# --- 3. Pipeline completa richiamata dal main --- #

def preprocess_pipeline(
    X_raw: pd.DataFrame,
    y_raw: pd.DataFrame,
    fixed_offset_hours: int = 10,
    debug: bool = True,
    save_processed: bool = False,
    output_dir: str = "data/processed",
):
    """
    Pipeline di preprocessing completa:
    1) fill missing (es. rain_1h)
    2) OHE weather_description con colonna 'other'
    3) timezone fix + cyclical encoding su X
    4) allineamento X-y
    5) cast numeriche a float32
    6) (opzionale) salvataggio su disco dei dataset processati
    """
    if debug:
        print("\n=== PREPROCESSING PIPELINE ===")
        print(f"[RAW] X: {X_raw.shape}, y: {y_raw.shape}")

    # 1) missing
    X = fill_missing_values(X_raw)

    # 2) OHE + other
    X = ohe_weather_description(X)

    # 3) timezone + cyc + align
    X_aligned, y_aligned = process_all_data(
        X, y_raw,
        x_time_col="dt_iso",
        y_time_col="datetime",
        fixed_offset_hours=fixed_offset_hours,
        debug=debug
    )

    # 4) cast a float32 delle colonne numeriche
    num_cols_X = X_aligned.select_dtypes(include=["number"]).columns
    num_cols_y = y_aligned.select_dtypes(include=["number"]).columns

    X_aligned[num_cols_X] = X_aligned[num_cols_X].astype("float32")
    y_aligned[num_cols_y] = y_aligned[num_cols_y].astype("float32")

    if debug:
        print("[DTYPES] X:")
        print(X_aligned.dtypes)
        print("[DTYPES] y:")
        print(y_aligned.dtypes)

    if save_processed:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        X_path = out_dir / "X_processed.csv"
        y_path = out_dir / "y_processed.csv"
        X_aligned.to_csv(X_path, index=True)
        y_aligned.to_csv(y_path, index=True)
        if debug:
            print(f"[SAVE] X -> {X_path}")
            print(f"[SAVE] y -> {y_path}")

    return X_aligned, y_aligned
