import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Import opzionali per analisi temporale avanzata
try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# =========================
# Utility per serie temporale della label
# =========================
def _extract_label_timeseries(
    y: pd.DataFrame,
    label_col: str,
    X_for_datetime: pd.DataFrame | None = None,
):
    """
    Ritorna una serie temporale pulita (datetime, valori numerici) per la label.

    Logica:
    - la label deve stare in y[label_col]
    - cerchiamo una colonna temporale tra y e X_for_datetime:
        1) prima per nome (datetime, dt_iso, date, time, ecc.)
        2) poi provando a convertire candidate a datetime e
           scegliendo quella con percentuale di valori validi più alta.

    Se non troviamo nulla, ritorniamo (None, None) e stampiamo un warning.
    """

    if label_col not in y.columns:
        print(f"[EDA] label '{label_col}' non trovata in y.columns: {list(y.columns)}")
        return None, None

    def _find_datetime_column(df: pd.DataFrame, df_name: str):
        if df is None:
            return None, None

        candidates_by_name = []
        # 1) per nome "classico"
        for c in df.columns:
            lower = c.lower()
            if any(k in lower for k in ["datetime", "dt_iso", "date", "time"]):
                candidates_by_name.append(c)

        # 2) se già datetime64
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                candidates_by_name.append(c)

        # 3) se non abbiamo ancora niente, consideriamo TUTTE le colonne come candidate
        if not candidates_by_name:
            candidates = list(df.columns)
        else:
            candidates = candidates_by_name

        best_col = None
        best_valid_ratio = 0.0
        best_series = None

        for c in candidates:
            try:
                dt_series = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                continue

            valid_ratio = (~dt_series.isna()).mean()
            # Se almeno metà dei valori sono parseabili, è plausibilmente una colonna tempo
            if valid_ratio > 0.5 and valid_ratio > best_valid_ratio:
                best_valid_ratio = valid_ratio
                best_col = c
                best_series = dt_series

        if best_col is not None:
            print(
                f"[EDA] Uso colonna temporale '{best_col}' da {df_name} "
                f"(valid_ratio={best_valid_ratio:.2f})"
            )
            return best_col, best_series

        return None, None

    # Prova prima in y, poi in X_for_datetime
    dt_col = None
    dt_series = None
    source = None

    dt_col, dt_series = _find_datetime_column(y, "y")
    if dt_col is not None:
        source = "y"
    elif X_for_datetime is not None:
        dt_col, dt_series = _find_datetime_column(X_for_datetime, "X")
        if dt_col is not None:
            source = "X"

    if dt_col is None or dt_series is None:
        print(
            "[EDA] Nessuna colonna temporale valida trovata né in y né in X; "
            "skip analisi temporale."
        )
        return None, None

    # Costruiamo la serie finale (datetime + label)
    # --> usiamo .to_numpy() per evitare che l'indice si porti dietro il nome 'datetime'
    label_values = pd.to_numeric(y[label_col], errors="coerce")

    ts = pd.DataFrame(
        {
            "datetime": dt_series.to_numpy(),
            label_col: label_values.to_numpy(),
        }
    )

    # Azzeriamo completamente l'indice per evitare ambiguità
    ts = ts.reset_index(drop=True)

    ts = ts.dropna(subset=["datetime", label_col]).sort_values("datetime")

    if ts.empty:
        print("[EDA] Serie temporale vuota dopo dropna; skip analisi temporale.")
        return None, None

    return ts["datetime"], ts[label_col]


def _plot_label_timeseries(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_col: str,
    output_dir: Path,
):
    dt, series = _extract_label_timeseries(y, label_col, X_for_datetime=X)
    if dt is None:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dt, series, lw=1.0, color="#55A868")
    ax.set_title(f"Andamento temporale di {label_col}")
    ax.set_xlabel("datetime")
    ax.set_ylabel(label_col)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / f"time_series_{label_col}.png", dpi=150)
    plt.close(fig)


def _plot_numeric_correlations(
    X: pd.DataFrame, y: pd.Series, output_dir: Path, label_col: str
):
    num_cols = X.select_dtypes(include=[np.number]).columns
    # escludi lat/lon
    num_cols = [c for c in num_cols if c.lower() not in {"lat", "lon"}]

    if len(num_cols) == 0:
        return

    corr_df = pd.concat([X[num_cols], y.rename(label_col)], axis=1)
    corr = corr_df.corr(numeric_only=True)
    if corr.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Matrice di correlazione (numeriche)")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="corr")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_numeric.png", dpi=150)
    plt.close(fig)


def _plot_numeric_distributions(X: pd.DataFrame, output_dir: Path):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return

    # Rimuove coordinate se presenti (restano comunque salvate altrove se servono)
    drop_cols = [c for c in numeric_cols if "lat" in c.lower() or "lon" in c.lower()]
    numeric_cols = [c for c in numeric_cols if c not in drop_cols]
    if len(numeric_cols) == 0:
        return

    # Salva stats base
    stats = X[numeric_cols].agg(["mean", "std"]).T
    stats.to_csv(output_dir / "numeric_stats.csv")

    for col in numeric_cols:
        series = pd.to_numeric(X[col], errors="coerce").dropna()
        if series.empty:
            continue

        mean_val = series.mean()
        std_val = series.std()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(series, bins=30, color="lightgreen", edgecolor="black", alpha=0.7)
        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"mean = {mean_val:.2f}",
        )
        ax.axvline(
            mean_val + std_val,
            color="orange",
            linestyle=":",
            linewidth=1.5,
            label=f"+1σ = {mean_val + std_val:.2f}",
        )
        ax.axvline(
            mean_val - std_val,
            color="orange",
            linestyle=":",
            linewidth=1.5,
            label=f"-1σ = {mean_val - std_val:.2f}",
        )
        ax.set_title(f"Distribuzione di {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequenza")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()
        fig.savefig(output_dir / f"hist_{col}.png", dpi=150)
        plt.close(fig)


def _plot_pv_vs_weather(
    X: pd.DataFrame, y: pd.DataFrame, output_dir: Path, label_col: str = "kwp"
):
    if "datetime" not in X.columns or "datetime" not in y.columns:
        return
    if "weather_description" not in X.columns:
        return
    merged = pd.merge(
        y[["datetime", label_col]] if label_col in y.columns else y,
        X[["datetime", "weather_description"]],
        on="datetime",
        how="inner",
    )
    merged["datetime"] = pd.to_datetime(merged["datetime"], errors="coerce")
    merged = merged.dropna(subset=["datetime", label_col])
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    data = merged.groupby("weather_description")[label_col]
    labels = list(data.groups.keys())
    values = [pd.to_numeric(data.get_group(k), errors="coerce") for k in labels]
    ax.boxplot(
        values,
        labels=labels,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="#D0E6A5", color="#3B5A3C"),
        medianprops=dict(color="#C13B00"),
    )
    ax.set_title("PV Power vs Weather Condition")
    ax.set_xlabel("weather_description")
    ax.set_ylabel(label_col)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "pv_vs_weather.png", dpi=150)
    plt.close(fig)


def _plot_categorical_distributions(X: pd.DataFrame, output_dir: Path):
    """
    Plot per le feature categoriche "vere":
    - esclude dt_iso / datetime
    - esclude colonne con troppe categorie (es. > 30 livelli) per evitare grafici illeggibili.
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    # Escludi esplicitamente colonne temporali usate come stringa
    exclude_names = {"dt_iso", "datetime"}
    cat_cols = [c for c in cat_cols if c.lower() not in exclude_names]

    for col in cat_cols:
        value_counts = X[col].value_counts()
        if value_counts.empty:
            continue

        # se ha troppe categorie, salta (tipico per ID, stringhe quasi univoche, ecc.)
        if value_counts.shape[0] > 30:
            continue

        percentages = value_counts / value_counts.sum() * 100

        # --- Barplot orizzontale migliorato ---
        fig_bar, ax_bar = plt.subplots(
            figsize=(10, max(4, 0.4 * len(value_counts)))
        )

        ax_bar.barh(
            value_counts.index,
            value_counts.values,
            color="steelblue"
        )

        ax_bar.set_title(f"Distribuzione di {col}")
        ax_bar.set_xlabel("Frequenza")
        ax_bar.set_ylabel(col)

        # Categoria più frequente in alto
        ax_bar.invert_yaxis()

        # Griglia leggera sull'asse x
        ax_bar.grid(axis="x", linestyle="--", alpha=0.7)

        fig_bar.tight_layout()
        fig_bar.savefig(output_dir / f"bar_{col}.png", dpi=150)
        plt.close(fig_bar)

        # Pie plot
        fig_pie, ax_pie = plt.subplots(figsize=(7, 5))
        wedges, _ = ax_pie.pie(
            value_counts,
            labels=None,
            colors=plt.cm.tab20(np.linspace(0, 1, len(value_counts))),
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )
        labels = [
            f"{cat}: {pct:.1f}%"
            for cat, pct in zip(value_counts.index, percentages)
        ]
        ax_pie.legend(
            wedges,
            labels,
            title="Categorie",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=9,
        )
        ax_pie.set_title(f"Distribuzione percentuale di {col}")
        fig_pie.subplots_adjust(right=0.7)
        fig_pie.savefig(output_dir / f"pie_{col}.png", dpi=150)
        plt.close(fig_pie)


# ====================================
#  STAGE 1 – Analisi temporale label
# ====================================
def _plot_acf_pacf_label(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_col: str,
    output_dir: Path,
    max_lag: int = 168,  # ad es. 1 settimana a risoluzione oraria
):
    """ACF e PACF della serie PV (se statsmodels è disponibile)."""
    if not _HAS_STATSMODELS:
        print("[WARN] statsmodels non disponibile: skip ACF/PACF.")
        return

    dt, series = _extract_label_timeseries(y, label_col, X_for_datetime=X)
    if dt is None:
        return

    series = series - series.mean()  # centriamo

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, ax=axes[0], lags=max_lag)
    axes[0].set_title(f"ACF di {label_col} (fino a {max_lag} lag)")
    plot_pacf(series, ax=axes[1], lags=min(max_lag, 40), method="ywm")
    axes[1].set_title(f"PACF di {label_col}")
    fig.tight_layout()
    fig.savefig(output_dir / f"acf_pacf_{label_col}.png", dpi=150)
    plt.close(fig)


def _plot_seasonal_decomposition_label(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_col: str,
    output_dir: Path,
    period: int = 24,  # 24 ore
):
    """Decomposizione stagionale (trend + stagionalità + residuo)."""
    if not _HAS_STATSMODELS:
        print("[WARN] statsmodels non disponibile: skip seasonal decomposition.")
        return

    dt, series = _extract_label_timeseries(y, label_col, X_for_datetime=X)
    if dt is None:
        return

    # La decomposizione richiede serie senza NaN
    series_clean = series.dropna()
    if series_clean.empty:
        return

    try:
        result = seasonal_decompose(series_clean, period=period, model="additive")
    except Exception as e:
        print(f"[WARN] Seasonal decomposition failed: {e}")
        return

    fig = result.plot()
    fig.set_size_inches(10, 8)
    fig.suptitle(f"Decomposizione stagionale di {label_col} (period={period})", y=0.95)
    fig.tight_layout()
    fig.savefig(output_dir / f"seasonal_decomp_{label_col}.png", dpi=150)
    plt.close(fig)


def _plot_daily_profile_label(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_col: str,
    output_dir: Path,
):
    """
    Profilo giornaliero medio: mediana + IQR per ogni ora del giorno.
    Utile per vedere la forma "tipica" del giorno fotovoltaico.
    """
    dt, series = _extract_label_timeseries(y, label_col, X_for_datetime=X)
    if dt is None:
        return

    df = pd.DataFrame({"datetime": dt, label_col: series})
    df["hour"] = df["datetime"].dt.hour

    grouped = df.groupby("hour")[label_col]
    median = grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)

    fig, ax = plt.subplots(figsize=(8, 4))
    hours = median.index.values
    ax.plot(hours, median.values, marker="o", label="Mediana", linewidth=2)
    ax.fill_between(hours, q25.values, q75.values, alpha=0.3, label="IQR (25°–75°)")
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Ora del giorno")
    ax.set_ylabel(label_col)
    ax.set_title(f"Profilo giornaliero medio di {label_col}")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"daily_profile_{label_col}.png", dpi=150)
    plt.close(fig)


def _plot_power_spectrum_label(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_col: str,
    output_dir: Path,
):
    """
    Spettro in frequenza della serie PV (FFT).
    Campionamento: 1 punto/ora -> frequenza in cicli per ora.
    """
    dt, series = _extract_label_timeseries(y, label_col, X_for_datetime=X)
    if dt is None:
        return

    series = series.dropna()
    if series.empty:
        return

    # Rimuoviamo la media per un segnale più stabile
    x = series.values - series.values.mean()
    n = len(x)
    # FFT a lato positivo
    fft_vals = np.fft.rfft(x)
    fft_freqs = np.fft.rfftfreq(n, d=1.0)  # d=1 ora

    power = (np.abs(fft_vals) ** 2) / n

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fft_freqs, power)
    ax.set_xlabel("Frequenza (cicli/ora)")
    ax.set_ylabel("Potenza")
    ax.set_title(f"Spettro di potenza di {label_col}")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / f"power_spectrum_{label_col}.png", dpi=150)
    plt.close(fig)


def run_temporal_analysis(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_col: str = "kwp",
    output_dir: str = "eda_plots/temporal",
):
    """
    Esegue l'analisi temporale della label:
    - ACF/PACF
    - Decomposizione stagionale (24h)
    - Profilo giornaliero medio
    - Spettro in frequenza
    """
    out_path = Path(output_dir)
    _ensure_dir(out_path)

    print("\n=== ANALISI TEMPORALE LABEL ===")

    _plot_acf_pacf_label(X, y, label_col, out_path)
    _plot_seasonal_decomposition_label(X, y, label_col, out_path, period=24)
    _plot_daily_profile_label(X, y, label_col, out_path)
    _plot_power_spectrum_label(X, y, label_col, out_path)

    print(f"[EDA] Plot temporali salvati in '{output_dir}'")


# =============================
#  Wrapper EDA esistente
# =============================
def generate_basic_plots(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_col: str = "kwp",
    output_dir: str = "eda_plots",
    detailed: bool = True,
):
    """
    Salva alcuni plot base (time-series label, corr numeriche, distribuzioni).
    """
    out_path = Path(output_dir)
    _ensure_dir(out_path)

    y_series = (
        pd.to_numeric(y[label_col], errors="coerce")
        if label_col in y.columns
        else None
    )

    if y_series is not None:
        _plot_label_timeseries(X, y, label_col, out_path)
        _plot_numeric_correlations(X, y_series, out_path, label_col)
    if detailed:
        label_for_weather = label_col if label_col in y.columns else y.columns[-1]
        _plot_numeric_distributions(X, out_path)
        _plot_pv_vs_weather(X, y, out_path, label_col=label_for_weather)
        _plot_categorical_distributions(X, out_path)


def run_basic_eda(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_col: str = "kwp",
    make_plots: bool = True,
    temporal_analysis: bool = True,
):
    """Stampe veloci per capire com'è il dataset + plot di base (+ opzionale analisi temporale)."""
    print("\n=== EDA BASE ===")
    print("[X] prime righe:")
    print(X.head())
    print("\n[y] prime righe:")
    print(y.head())

    print("\n[X] missing per colonna:")
    print(X.isnull().sum())

    print("\n[y] missing per colonna:")
    print(y.isnull().sum())

    if make_plots:
        generate_basic_plots(X, y, label_col=label_col)
        print("\n[EDA] Plot base salvati in 'eda_plots/'")

        if temporal_analysis:
            run_temporal_analysis(
                X,
                y,
                label_col=label_col,
                output_dir="eda_plots/temporal",
            )


def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """Correlation ratio η per categoriali vs numerica."""
    df = pd.DataFrame({"cat": categories, "y": measurements}).dropna()
    if df.empty or df["cat"].nunique() <= 1:
        return np.nan

    y = df["y"].astype(float).values
    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum()
    if ss_tot == 0:
        return np.nan

    ss_between = 0.0
    for _, g in df.groupby("cat")["y"]:
        n = len(g)
        mu = g.mean()
        ss_between += n * (mu - y_mean) ** 2

    eta2 = ss_between / ss_tot
    return np.sqrt(eta2)


def analyze_feature_label_correlations(
    X: pd.DataFrame,
    y: pd.DataFrame,
    label_col: str = "kwp",
    top_k: int = 15,
):
    """Analizza correlazioni tra feature e label (Pearson + η)."""
    print("\n=== ANALISI CORRELAZIONI CON LA LABEL ===")

    if label_col not in y.columns:
        raise ValueError(f"Colonna label '{label_col}' non trovata in y.")

    y_series = pd.to_numeric(y[label_col], errors="coerce")

    # ---- escludiamo le colonne temporali (dt_iso, datetime, ecc.) ----
    exclude_cols = set()
    for col in X.columns:
        name = col.lower()
        if name in {"dt_iso", "datetime"}:
            exclude_cols.add(col)
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            exclude_cols.add(col)

    # ---- escludiamo lat / lon (costanti e inutili) ----
    for col in X.columns:
        if col.lower() in {"lat", "lon"}:
            exclude_cols.add(col)

    X_filtered = X.drop(columns=exclude_cols, errors="ignore")

    # Ora selezioniamo numeriche e categoriche SENZA le categorie che non ci interessano
    X_num = X_filtered.select_dtypes(include=[np.number])
    X_cat = X_filtered.select_dtypes(include=["object", "category"])

    # Pearson per numeriche
    pearson = {
        col: X_num[col].corr(y_series)
        for col in X_num.columns
        if X_num[col].nunique() > 1
    }
    pearson_s = pd.Series(pearson, name="correlation")

    # Correlation ratio per categoriali
    eta = {col: correlation_ratio(X_cat[col], y_series) for col in X_cat.columns}
    eta_s = pd.Series(eta, name="correlation")

    corr_df = pd.concat([
        pearson_s.to_frame().assign(Type="Numerica"),
        eta_s.to_frame().assign(Type="Categorica"),
    ]).dropna()

    corr_df = corr_df.reindex(
        corr_df["correlation"].abs().sort_values(ascending=False).index
    )

    print(f"\nTop {top_k} feature più correlate con la label:")
    print(corr_df.head(top_k))