import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _plot_label_timeseries(y: pd.DataFrame, label_col: str, output_dir: Path):
    if "datetime" not in y.columns or label_col not in y.columns:
        return
    ts = y.copy()
    ts["datetime"] = pd.to_datetime(ts["datetime"], errors="coerce")
    ts = ts.dropna(subset=["datetime", label_col]).sort_values("datetime")
    if ts.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        ts["datetime"],
        pd.to_numeric(ts[label_col], errors="coerce"),
        lw=1.0,
        color="#55A868",
    )
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

        # Bar plot
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
        _plot_label_timeseries(y, label_col, out_path)
        _plot_numeric_correlations(X, y_series, out_path, label_col)
    if detailed:
        label_for_weather = label_col if label_col in y.columns else y.columns[-1]
        _plot_numeric_distributions(X, out_path)
        _plot_pv_vs_weather(X, y, out_path, label_col=label_for_weather)
        _plot_categorical_distributions(X, out_path)


def run_basic_eda(X: pd.DataFrame, y: pd.DataFrame, label_col: str = "kwp", make_plots: bool = True):
    """Stampe veloci per capire com'è il dataset."""
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
        print("\n[EDA] Plot salvati in 'eda_plots/'")

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

    '''# Rimuoviamo tutte le colonne temporali da X
    datetime_cols = [
    col for col in X.columns
    if col.lower() in ["dt_iso", "datetime"]
    or "date" in col.lower()
    or "time" in col.lower()
    or str(X[col].dtype).startswith("datetime")
    ]
    X_filtered = X.drop(columns=datetime_cols, errors="ignore")'''

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
