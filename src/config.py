# src/config.py

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal, Mapping, Tuple
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
    # nomi/pattern per artifact salvati (personalizzabili per modelli diversi)
    model_filename: str = "model_tcn.pth"
    model_fold_template: str = "model_tcn_fold{fold}.pth"
    scaler_filename: str = "scaler.pkl"
    scaler_fold_template: str = "scaler_fold{fold}.pkl"
    X_val_filename: str = "X_val_scaled.npy"
    y_val_filename: str = "y_val_scaled.npy"
    X_val_fold_template: str = "X_val_scaled_fold{fold}.npy"
    y_val_fold_template: str = "y_val_scaled_fold{fold}.npy"
    y_train_filename: str = "y_train_scaled.csv"
    y_train_fold_template: str = "y_train_scaled_fold{fold}.csv"
    y_val_out_filename: str = "y_val_scaled.csv"
    y_val_out_fold_template: str = "y_val_scaled_fold{fold}.csv"

    # salvataggi feature-engineered (opzionali)
    X_train_feat_out: str = "data/processed/X_train_feat.csv"
    X_val_feat_out: str = "data/processed/X_val_feat.csv"


# -----------------------------
# Config dello split
# -----------------------------

@dataclass
class SplitConfig:
    # "train_val" semplice oppure "cv"
    mode: Literal["train_val", "cv"] = "cv"
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

### PRIMA MODIFICA PER TUNING ###
@dataclass
class ModelConfig:
    # Identificatore architettura (usato da build_model e random_search)
    arch: str = "tcn"   # es: "tcn", "lstm_mlp", ...

    # Dimensioni (settate nel main via loader/dataset)
    input_size: Optional[int] = None
    horizon: Optional[int] = None

    # -----------------
    # TCN params (legacy)
    # -----------------
    # modifica per tuning
    num_channels: Tuple[int, ...] = (64, 64, 64)
    hidden: int = 64
    kernel_size: int = 2 # prima era 3
    n_blocks: int = 3
    dropout: float = 0.2984182840729522 # 0.2

    # -----------------
    # LSTM+MLP params
    # -----------------
    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = False
    mlp_hidden_size: int = 64



@dataclass
class TrainingConfig:
    epochs: int = 30
    #lr: float = 1e-3
    lr: float = 0.0025652733300618224
    # modifica per tuning
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.0
    loss_plot_path: str = "eda_plots/loss_curve.png"
    pred_vs_true_plot_path: str = "eda_plots/pred_vs_true.png"


SearchSpace = Mapping[str, Any]

@dataclass
class RandomSearchConfig:
    enabled: bool = True
    n_trials: int = 20
    metric: Literal["loss", "rmse", "mase"] = "loss"
    mode: Literal["min", "max"] = "min"
    seed: int = 42
    # se vuoi silenziare il log per ogni trial
    verbose: bool = True

    # spazio di ricerca agnostico (path -> spec)
    # NB: tuple ok (es. num_channels)
    search_space: Dict[str, Any] = field(default_factory=lambda: {
        "model.num_channels": [
            (16, 16),
            (32, 32),
            (32, 32, 32),
            (64, 64, 64),
        ],
        "model.kernel_size": {"type": "int", "low": 2, "high": 5},
        "model.dropout": {"type": "float", "low": 0.05, "high": 0.5},
        "training.lr": {"type": "logfloat", "low": 1e-4, "high": 5e-3},
        # opzionale: fai cercare anche epochs (ma occhio ai tempi)
        # "training.epochs": {"type":"int","low":5,"high":30},
    })


# -----------------------------
# Config baseline ML
# -----------------------------

@dataclass
class BaselineConfig:
    enabled: bool = True
    ridge_alpha: float = 1.0
    mase_seasonal_m: int = 24


# -----------------------------
# Config "alto livello" di tutto l’esperimento
# -----------------------------

@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    data: PVDataConfig = field(
        default_factory=lambda: PVDataConfig(
            history_hours=72,
            horizon_hours=24,
            stride=1,
            include_future_covariates=False,
            include_past_target=True,
        )
    )
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    random_search: RandomSearchConfig = field(default_factory=RandomSearchConfig)

    # Search space agnostico: puoi aggiungere nuove arch in futuro
    hyperparam_search: Dict[str, Any] = field(
        default_factory=lambda: {
            "tcn": {
                # modifica per tuning
                "model.num_channels": [
                    (16, 16),
                    (32, 32),
                    (32, 32, 32),
                    (64, 64, 64),
                ],
                "model.dropout": {"type": "float", "low": 0.0, "high": 0.5},
                "training.lr": {"type": "logfloat", "low": 1e-4, "high": 3e-3},
                "training.epochs": {"type": "int", "low": 5, "high": 30},
            },
            "lstm_mlp": {
                "model.hidden_size": {"type": "int", "low": 32, "high": 256, "step": 32},
                "model.num_layers": {"type": "int", "low": 1, "high": 4},
                "model.dropout": {"type": "float", "low": 0.0, "high": 0.6},
                "model.mlp_hidden_size": {"type": "int", "low": 32, "high": 256, "step": 32},
                "training.lr": {"type": "logfloat", "low": 1e-4, "high": 3e-3},
                "training.epochs": {"type": "int", "low": 5, "high": 30},
            },
        }
    )
