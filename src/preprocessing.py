import numpy as np
import pandas as pd
from datetime import timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# --- 1. Funzioni semplici: missing --- #

DEFAULT_RARE_WEATHER = [
    "haze", "light intensity shower rain", "heavy intensity rain",
    "shower rain", "light intensity drizzle", "mist", "smoke",
    "fog", "thunderstorm", "thunderstorm with rain",
    "drizzle", "proximity squalls"
]


def fill_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """Imputazioni deterministiche (es. rain_1h → 0)."""
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


def drop_site_coords(X: pd.DataFrame) -> pd.DataFrame:
    """Rimuove lat/lon (costanti) dopo averli estratti."""
    return X.drop(columns=[c for c in ["lat", "lon"] if c in X.columns], errors="ignore")


# --- 2. Funzioni OHE train/transform --- #

def _replace_rare_weather(X: pd.DataFrame, rare_list: List[str]) -> pd.DataFrame:
    X = X.copy()
    if "weather_description" in X.columns:
        X["weather_description"] = X["weather_description"].replace(rare_list, "other")
    return X


def fit_ohe_on_train(
    X_train: pd.DataFrame,
    rare_list: List[str] = DEFAULT_RARE_WEATHER,
) -> tuple[pd.DataFrame, List[str]]:
    """
    Fit OHE solo su train e restituisce le colonne finali (vocabolario fissato).
    """
    X_train = _replace_rare_weather(X_train, rare_list)
    if "weather_description" in X_train.columns:
        X_train = pd.get_dummies(
            X_train,
            columns=["weather_description"],
            prefix="weather",
            dummy_na=False,
            dtype=int,
        )
    X_train = drop_site_coords(X_train)
    dummy_columns = list(X_train.columns)
    return X_train, dummy_columns


def transform_ohe_with_vocab(
    X: pd.DataFrame,
    dummy_columns: List[str],
    rare_list: List[str] = DEFAULT_RARE_WEATHER,
) -> pd.DataFrame:
    """
    Applica OHE usando il vocabolario del train (colonne forzate e ordinate).
    """
    X = _replace_rare_weather(X, rare_list)
    if "weather_description" in X.columns:
        X = pd.get_dummies(
            X,
            columns=["weather_description"],
            prefix="weather",
            dummy_na=False,
            dtype=int,
        )
    X = drop_site_coords(X)
    X = X.reindex(columns=dummy_columns, fill_value=0)
    return X


# --- 3. Funzioni di scaling --- #

import numpy as np
import pandas as pd
from typing import Dict, Union


# ===============================================================
# FIT SCALER (solo su TRAIN)
# ===============================================================
def fit_scaler_on_train(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray, list],
    mode: str | None = None,
) -> Dict:
    """
    Calcola statistiche di scaling SOLO sul training, sia per X che per y.
    
    Parametri:
        X_train: DataFrame delle feature di train
        y_train: array/Series con i target di train (shape [N] o [N,H])
        mode: None | "standard" | "minmax"

    Ritorna:
        dict con stats di X e y
    """
    if mode is None:
        return {}

    if mode not in {"standard", "minmax"}:
        raise ValueError("mode deve essere None, 'standard', 'minmax'.")

    scaler = {"mode": mode, "X_stats": {}, "y_stats": {}}

    # ---------- 1) Scaling delle feature X ----------
    float_cols = X_train.select_dtypes(include=["float", "float32", "float64"]).columns

    for col in float_cols:
        col_values = X_train[col].to_numpy(dtype=np.float32)
        if mode == "standard":
            mean = float(np.mean(col_values))
            std = float(np.std(col_values) + 1e-8)
            scaler["X_stats"][col] = {"mean": mean, "std": std}
        else:  # minmax
            cmin = float(np.min(col_values))
            cmax = float(np.max(col_values))
            scaler["X_stats"][col] = {"min": cmin, "max": cmax}

    # ---------- 2) Scaling del target y ----------
    y_arr = np.asarray(y_train, dtype=np.float32)

    # se y è many-to-one: [N]; se many-to-many: [N, H]
    if mode == "standard":
        scaler["y_stats"]["mean"] = float(np.mean(y_arr))
        scaler["y_stats"]["std"] = float(np.std(y_arr) + 1e-8)
    else:
        scaler["y_stats"]["min"] = float(np.min(y_arr))
        scaler["y_stats"]["max"] = float(np.max(y_arr))

    return scaler


# ===============================================================
# APPLY SCALER (usa params del train, vale per val/test)
# ===============================================================
def apply_scaler(
    X_or_y: Union[pd.DataFrame, pd.Series, np.ndarray],
    scaler: Dict,
    is_target: bool = False,
):
    """
    Applica scaling a X oppure y usando le statistiche salvate
    (calcolate SOLO sul train).

    Parametri:
        X_or_y: DataFrame (X) o array/Series (y)
        scaler: dizionario uscito da fit_scaler_on_train
        is_target: se True, applica scaling della y (non guarda colonne)

    Ritorna:
        X_scaled (DataFrame) oppure y_scaled (array)
    """
    if not scaler:
        return X_or_y

    mode = scaler["mode"]

    # ================================================================
    # 1) Scaling della y
    # ================================================================
    if is_target:
        y_arr = np.asarray(X_or_y, dtype=np.float32)
        stats = scaler["y_stats"]

        if mode == "standard":
            y_arr = (y_arr - stats["mean"]) / stats["std"]
        else:
            denom = (stats["max"] - stats["min"]) or 1e-8
            y_arr = (y_arr - stats["min"]) / denom

        y_arr = y_arr.astype(np.float32)

        # 🔴 QUI: preserviamo il tipo originale
        if isinstance(X_or_y, pd.DataFrame):
            y_scaled = X_or_y.copy()
            # funziona sia per (N,1) che per (N,H)
            y_scaled.iloc[:, :] = y_arr
            return y_scaled

        elif isinstance(X_or_y, pd.Series):
            y_scaled = X_or_y.copy()
            y_scaled.iloc[:] = y_arr.reshape(-1)
            return y_scaled

        else:
            # se era array, lasciamo array
            return y_arr

    # ================================================================
    # 2) Scaling di X
    # ================================================================
    X_scaled = X_or_y.copy()
    X_stats = scaler["X_stats"]

    for col, params in X_stats.items():
        if col not in X_scaled.columns:
            continue

        arr = X_scaled[col].to_numpy(dtype=np.float32)

        if mode == "standard":
            arr = (arr - params["mean"]) / params["std"]
        else:
            denom = (params["max"] - params["min"]) or 1e-8
            arr = (arr - params["min"]) / denom

        X_scaled[col] = arr.astype(np.float32)

    return X_scaled

# --- 4. Funzione principale: timezone + cyclical encoding + align --- #

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


# --- 5. Pipeline base (deterministica) richiamata dal main --- #

def preprocess_pipeline(
    X_raw: pd.DataFrame,
    y_raw: pd.DataFrame,
    fixed_offset_hours: int = 10,
    debug: bool = True,
    save_processed: bool = False,
    output_dir: str = "data/processed",
):
    """
    Pipeline deterministica (niente fit su val/test):
    1) fill missing (es. rain_1h)
    2) timezone fix + cyclical encoding su X
    3) allineamento X-y
    4) cast numeriche a float32
    5) (opzionale) salvataggio su disco dei dataset processati

    L'OHE e lo scaling vanno fittati a parte sul train.
    """
    if debug:
        print("\n=== PREPROCESSING PIPELINE (deterministica) ===")
        print(f"[RAW] X: {X_raw.shape}, y: {y_raw.shape}")

    # 1) missing
    X = fill_missing_values(X_raw)

    # 2) timezone + cyc + align
    X_aligned, y_aligned = process_all_data(
        X, y_raw,
        x_time_col="dt_iso",
        y_time_col="datetime",
        fixed_offset_hours=fixed_offset_hours,
        debug=debug
    )

    # 3) cast a float32 delle colonne numeriche
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