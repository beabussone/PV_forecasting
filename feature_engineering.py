import numpy as np
import pandas as pd
from numpy import sin, cos, tan, radians, degrees
from pathlib import Path


# ============================================================
# 1.  Solar geometry component building blocks
# ============================================================

def solar_declination(day_of_year: int) -> float:
    """
    Declinazione solare in gradi.
    Formula NREL / Duffie-Beckman.
    """
    return 23.45 * sin(radians(360 * (284 + day_of_year) / 365))


def hour_angle(local_solar_time_hours: float) -> float:
    """
    Hour Angle H in gradi.
    0° = mezzogiorno solare, negativo al mattino, positivo al pomeriggio.
    """
    return 15 * (local_solar_time_hours - 12)


def solar_elevation_angle(lat: float, dec: float, hra: float) -> float:
    """
    Angolo di elevazione solare (0° = orizzonte, 90° = zenith).
    """
    return degrees(np.arcsin(
        sin(radians(lat)) * sin(radians(dec)) +
        cos(radians(lat)) * cos(radians(dec)) * cos(radians(hra))
    ))


def solar_azimuth_angle(lat: float, dec: float, hra: float) -> float:
    """
    Angolo di azimuth del sole (0° = Nord, 90° = Est, 180° = Sud, 270° = Ovest).
    Formula NREL.
    """
    az = degrees(np.arctan2(
        -sin(radians(hra)),
        cos(radians(hra)) * sin(radians(lat)) - tan(radians(dec)) * cos(radians(lat))
    ))
    return (az + 360) % 360   # normalizzato in [0, 360]


def extraterrestrial_irradiance(day_of_year: int, lat: float, dec: float, hra: float):
    """
    Irradianza solare extraterrestre (senza atmosfera).
    ETR = I_sc * E_c * cos(theta_z)
    """
    I_sc = 1367  # constante solare W/m²
    E_c = 1 + 0.033 * cos(radians(360 * day_of_year / 365))

    cos_theta = (
        sin(radians(lat)) * sin(radians(dec)) +
        cos(radians(lat)) * cos(radians(dec)) * cos(radians(hra))
    )

    # la componente negativa non ha significato fisico
    return I_sc * E_c * np.maximum(cos_theta, 0)


# ============================================================
# 2. Feature engineering alto livello
# ============================================================

def add_solar_features(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    dt_col: str = "dt_iso",
    ghi_col: str = "Ghi",
):
    """
    Aggiunge SOLO tre feature:
    - solar_zenith      (in gradi)
    - solar_azimuth     (in gradi)
    - clearness_index   (adimensionale)

    Usa il datetime come:
    - df[dt_col] se esiste la colonna
    - altrimenti l'indice, se è un DatetimeIndex.

    Non lascia variabili intermedie nel dataframe.
    """

    df = df.copy()

    # ---- 1) Ricaviamo la serie temporale dt ----
    if dt_col in df.columns:
        dt = pd.to_datetime(df[dt_col])
    else:
        # caso X_proc: datetime è nell'indice
        if isinstance(df.index, pd.DatetimeIndex):
            dt = df.index
        else:
            raise KeyError(
                f"Impossibile trovare una colonna datetime '{dt_col}' "
                "e l'indice non è un DatetimeIndex."
            )

    if ghi_col not in df.columns:
        raise KeyError(f"Colonna GHI '{ghi_col}' non trovata in df.")

    # ---- 2) Componenti temporali (SOLO variabili temporanee) ----
    day_of_year = dt.dayofyear.astype(int)
    hour_decimal = dt.hour + dt.minute / 60.0

    # --- Parametri astronomici (senza apply su Index) ---
    declination = np.array([solar_declination(int(d)) for d in day_of_year])
    hour_ang    = np.array([hour_angle(h)            for h in hour_decimal])

    # ---- 3) Angoli solari ----
    # Elevation → da cui ricaviamo lo zenith
    solar_elev = [
        solar_elevation_angle(lat, dec, hra)
        for dec, hra in zip(declination, hour_ang)
    ]
    # Zenith = 90° - elevazione
    solar_zenith = 90.0 - np.array(solar_elev)

    solar_azimuth = [
        solar_azimuth_angle(lat, dec, hra)
        for dec, hra in zip(declination, hour_ang)
    ]

    # ---- 4) Irradianza extraterrestre (ETR) solo come variabile intermedia ----
    etr = np.array([
        extraterrestrial_irradiance(int(day), lat, dec, hra)
        for day, dec, hra in zip(day_of_year, declination, hour_ang)
    ])

    # ---- 5) Clearness Index KI = GHI / ETR ----
    ghi = df[ghi_col].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        ki = ghi / np.where(etr <= 0, np.nan, etr)

    # bounding per togliere outlier assurdi
    ki = np.clip(ki, 0, 1.5)

    # ---- 6) Scriviamo SOLO le 3 feature finali nel df ----
    df["solar_zenith"] = solar_zenith.astype("float32")
    df["solar_azimuth"] = np.array(solar_azimuth, dtype="float32")
    df["clearness_index"] = ki.astype("float32")

    return df

# ============================================================
# 3. Alternative-to-POA physically-based features
# ============================================================

def add_effective_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge 3 feature fisiche che NON richiedono tilt:
    - effective_irradiance   = DNI*cos(zenith) + DHI
    - direct_fraction         = DNI / (DNI + DHI)
    - clear_sky_index         = GHI / GHI_clear   (stima semplice)
    
    Assunzioni:
    - solar_zenith già presente nel df (in gradi)
    - colonne DHI, DNI, GHI presenti
    """

    df = df.copy()

    # --- Check colonne ---
    for col in ["Dhi", "Dni", "Ghi", "solar_zenith"]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing for effective features")

    # estrazione dati base
    dni = df["Dni"].to_numpy(dtype="float32")
    dhi = df["Dhi"].to_numpy(dtype="float32")
    ghi = df["Ghi"].to_numpy(dtype="float32")
    zenith_deg = df["solar_zenith"].to_numpy(dtype="float32")

    # conversione in radianti
    zenith_rad = np.radians(zenith_deg)

    # =====================================================
    # 1) effective irradiance (senza tilt)
    # =====================================================
    # proiezione della componente diretta + componente diffusa
    eff = dni * np.cos(zenith_rad) + dhi
    eff = np.clip(eff, 0, None)  # niente valori negativi

    df["effective_irradiance"] = eff.astype("float32")

    # =====================================================
    # 2) direct fraction
    # =====================================================
    with np.errstate(divide="ignore", invalid="ignore"):
        direct_frac = dni / (dni + dhi)
    direct_frac = np.nan_to_num(direct_frac, nan=0.0, posinf=1.0, neginf=0.0)

    df["direct_fraction"] = direct_frac.astype("float32")

    # =====================================================
    # 3) clear-sky index (semplificato)
    # =====================================================
    # modello semplice per GHI_clear:
    # GHI_clear = ghi_potenziale = k * cos(zenith)
    # dove k = irradiance extraterrestre media ≈ 1367*(1+0.033 cos(...)) ~ 1000 W/m² scalati
    # per robustezza:
    
    ghi_clear = 1000 * np.cos(zenith_rad)
    ghi_clear = np.clip(ghi_clear, 1e-6, None)  # evitare divisioni zero

    csi = ghi / ghi_clear
    csi = np.clip(csi, 0, 2.0)

    df["clear_sky_index"] = csi.astype("float32")

    return df



def save_feature_engineered_X(X_feat: pd.DataFrame, out_path: str = "data/processed/X_feat.csv"):
    """
    Salva il dataframe X_feat in un percorso specificato.
    Crea automaticamente le cartelle se non esistono.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    X_feat.to_csv(out, index=True)
    print(f"[SAVE] X_feat -> {out}")