# src/config.py

from dataclasses import dataclass, field
from typing import Literal

from src.data_module import PVDataConfig
from src.models import ModelConfig
from src.training import TrainingConfig


# -----------------------------
# Path dei file
# -----------------------------

@dataclass
class PathsConfig:
    wx_path: str = "data/wx_dataset.xlsx"
    pv_path: str = "data/pv_dataset.xlsx"

    # salvataggi feature-engineered (opzionali)
    X_train_feat_out: str = "data/processed/X_train_feat.csv"
    X_val_feat_out: str = "data/processed/X_val_feat.csv"
    X_test_feat_out: str = "data/processed/X_test_feat.csv"


# -----------------------------
# Config dello split
# -----------------------------

@dataclass
class SplitConfig:
    # "train_val_test" oppure "cv"
    mode: Literal["train_val_test", "cv"] = "train_val_test"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    n_splits: int = 5  # usato solo se mode == "cv"


# -----------------------------
# Config dei DataLoader + scaling
# -----------------------------

@dataclass
class DataloaderConfig:
    batch_size: int = 64
    num_workers: int = 0
    scaling_mode: str = "standard"  # usato dalla pipeline di preprocessing/scaling


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
            history_hours=168, #proviamo 7giorni (migliore), 5giorni, altrimenti 72ore
            horizon_hours=24,
            stride = 1,
            include_future_covariates=False,
        )
    )

    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)

    # modello: model_name verrà usato per scegliere LSTM/GRU,
    # input_size e horizon li sovrascriviamo a runtime in base ai dati
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            model_name="gru",
            input_size=1,     # placeholder, aggiornato nel main
            horizon=24,       # placeholder
            hidden_size=32,     #16 per GRU MASE=0.95 #32 provato con conv2d, 32 con GRU MASE=0.92
            num_layers=2,
            dropout=0.3, # 0.3 bene per GRU
        )
    )

    #iperparametri di training
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            epochs=50,
            lr=5e-4,
            weight_decay=1e-4,
            patience=5,
            min_delta=0.0,
        )
    )