# src/config.py

from dataclasses import dataclass, field
from typing import Optional, Literal

from src.data_module import PVDataConfig

# -----------------------------
# Path dei file
# -----------------------------

@dataclass
class PathsConfig:
    wx_path: str = "data/wx_dataset.xlsx"
    pv_path: str = "data/pv_dataset.xlsx"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"

    # salvataggi feature-engineered (opzionali)
    X_train_feat_out: str = "data/processed/X_train_feat.csv"
    X_val_feat_out: str = "data/processed/X_val_feat.csv"


# -----------------------------
# Config dello split
# -----------------------------

@dataclass
class SplitConfig:
    # "train_val" semplice oppure "cv"
    mode: Literal["train_val", "cv"] = "train_val"
    train_ratio: float = 0.8  # usato se mode == "train_val"
    val_ratio: float = 0.2    # usato se mode == "train_val"
    n_splits: int = 4         # usato se mode == "cv"


# -----------------------------
# Config dei DataLoader + scaling
# -----------------------------

@dataclass
class DataloaderConfig:
    batch_size: int = 64
    num_workers: int = 0
    scaling_mode: str = "standard"  # usato dalla pipeline di preprocessing/scaling


# -----------------------------
# Config del modello e del training
# -----------------------------

@dataclass
class ModelConfig:
    hidden: int = 64
    kernel_size: int = 3
    n_blocks: int = 3
    dropout: float = 0.5
    input_size: Optional[int] = None
    horizon: Optional[int] = None


@dataclass
class TrainingConfig:
    epochs: int = 10
    lr: float = 1e-3
    loss_plot_path: str = "eda_plots/loss_curve.png"
    pred_vs_true_plot_path: str = "eda_plots/pred_vs_true.png"


# -----------------------------
# Config "alto livello" di tutto l’esperimento
# -----------------------------

@dataclass
class ExperimentConfig:
    # usare SEMPRE default_factory per oggetti mutabili / dataclass annidate

    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    # configurazione del dataset PV (finestra storica + orizzonte)
    data: PVDataConfig = field(
        default_factory=lambda: PVDataConfig(
            history_hours=168,
            horizon_hours=24,
            stride = 1,
            include_future_covariates=False,
        )
    )

    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
