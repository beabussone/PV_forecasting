import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ============================================================
# Config & Dataset
# ============================================================

@dataclass
class PVDataConfig:
    """
    Configurazione base per il dataset PV.
    - history_hours: numero di ore di storia in input (finestra passata)
    - horizon_hours: numero di ore future da prevedere (multi-step)
    - include_future_covariates: se True, restituisce anche X_future
      (le covariate future sullo stesso orizzonte della label)
    - stride: passo della sliding window (default 1)
    - include_past_target: se True, concatena la y passata (kwp)
      dentro x_hist per forecasting autoregressivo senza leakage.
    """
    history_hours: int = 72
    horizon_hours: int = 24
    include_future_covariates: bool = False
    stride: int = 1
    include_past_target: bool = True


class PVForecastDataset(Dataset):
    """
    Genera finestre storiche (history) e target multistep (horizon) da serie orarie.
    Opzionalmente restituisce covariate future se disponibili nel train.

    PATCH: opzionalmente concatena la y passata (kwp fino a t) dentro x_hist
           per fare forecasting autoregressivo senza leakage.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        config: Optional[PVDataConfig] = None,
    ):
        if config is None:
            config = PVDataConfig()
        self.config = config

        if len(X) != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza.")

        # ordinamento temporale
        self.X = X.sort_index()
        self.y = y.sort_index()

        self.history = config.history_hours
        self.horizon = config.horizon_hours
        self.stride = config.stride
        self.include_future_covariates = config.include_future_covariates

        # 🔽 nuovo flag (non rompe nulla: se non esiste nel config → False)
        self.include_past_target = bool(getattr(config, "include_past_target", False))

        if self.stride <= 0:
            raise ValueError("stride deve essere >= 1.")

        # array numpy
        self.X_values = self.X.to_numpy(dtype=np.float32)
        # assumiamo che la prima colonna di y sia la label kwp
        self.y_values = self.y.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1)

        # numero massimo di indici di partenza validi
        max_start = len(self.X_values) - (self.history + self.horizon) + 1
        if max_start <= 0:
            raise ValueError("Serie troppo corta rispetto a history+horizon.")
        self.start_indices = list(range(0, max_start, self.stride))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.start_indices):
            raise IndexError("Indice fuori range per il dataset.")

        h_start = self.start_indices[idx]
        h_end = h_start + self.history
        f_end = h_end + self.horizon

        x_hist = self.X_values[h_start:h_end]           # (H, F)
        y_future = self.y_values[h_end:f_end]           # (K,)

        # ✅ PATCH: aggiungo la y passata (kwp) alle feature storiche
        # y_past copre gli stessi istanti di x_hist: [h_start, ..., h_end-1]
        if self.include_past_target:
            y_past = self.y_values[h_start:h_end].astype(np.float32).reshape(-1, 1)  # (H, 1)
            x_hist = np.concatenate([x_hist, y_past], axis=1)                        # (H, F+1)

        sample = {
            "x_hist": torch.from_numpy(x_hist),
            "y_future": torch.from_numpy(y_future),
        }

        if self.include_future_covariates:
            x_future = self.X_values[h_end:f_end]
            sample["x_future"] = torch.from_numpy(x_future)

        return sample


# ============================================================
# Split: solo Train / Val temporale
# ============================================================

def temporal_train_val_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split cronologico coerente (niente shuffle) tra train e val.
    Le serie devono avere stesso indice e lunghezza.

    train_ratio + val_ratio deve essere ~1.0.
    """
    if len(X) != len(y):
        raise ValueError("X e y devono avere stessa lunghezza per lo split temporale.")

    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio e val_ratio devono essere positivi.")

    if abs((train_ratio + val_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio deve essere uguale a 1.0.")

    X_sorted = X.sort_index()
    y_sorted = y.sort_index()

    n = len(X_sorted)
    train_end = int(n * train_ratio)

    if train_end <= 0 or train_end >= n:
        raise ValueError("train_ratio produce uno split vuoto.")

    X_train, y_train = X_sorted.iloc[:train_end], y_sorted.iloc[:train_end]
    X_val, y_val = X_sorted.iloc[train_end:], y_sorted.iloc[train_end:]

    return X_train, X_val, y_train, y_val


# ============================================================
# Split: Cross-Validation temporale (solo train/val)
# ============================================================

def temporal_cv_splits_train_val(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 5,
    val_size: Optional[int] = None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Genera split di Cross-Validation per serie temporali senza test finale.
    - train = finestra che cresce progressivamente
    - val   = blocco subito successivo al train (dimensione fissa)

    Restituisce lista di tuple (X_train, X_val, y_train, y_val).
    """
    if len(X) != len(y):
        raise ValueError("X e y devono avere stessa lunghezza per la CV temporale.")

    if n_splits < 2:
        raise ValueError("n_splits deve essere almeno 2.")

    X_sorted = X.sort_index()
    y_sorted = y.sort_index()
    n = len(X_sorted)

    if val_size is None:
        # (n_splits blocchi val) + train rimanente
        val_size = n // (n_splits + 1)
        if val_size == 0:
            raise ValueError("Serie troppo corta rispetto a n_splits/val_size.")

    splits: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []

    for k in range(n_splits):
        train_end = val_size * (k + 1)
        val_start = train_end
        val_end = val_start + val_size

        if val_end > n:
            break

        X_train = X_sorted.iloc[:train_end]
        y_train = y_sorted.iloc[:train_end]
        X_val = X_sorted.iloc[val_start:val_end]
        y_val = y_sorted.iloc[val_start:val_end]

        splits.append((X_train, X_val, y_train, y_val))

    if not splits:
        raise ValueError("Nessuno split CV generato: controlla n_splits/val_size/len(X).")

    return splits


# ============================================================
# DataLoader helper
# ============================================================

def build_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Costruisce un DataLoader standard.
    Per serie temporali di solito:
    - train: shuffle=True (mescola le finestre)
    - val/test: shuffle=False
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
