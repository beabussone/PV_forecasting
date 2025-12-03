import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict


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
    """
    history_hours: int = 72
    horizon_hours: int = 24
    include_future_covariates: bool = False
    stride: int = 1


class PVForecastDataset(Dataset):
    """
    Genera finestre storiche (history) e target multistep (horizon) da serie orarie.
    Opzionalmente restituisce covariate future se disponibili nel train.
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

        x_hist = self.X_values[h_start:h_end]
        y_future = self.y_values[h_end:f_end]

        sample = {
            "x_hist": torch.from_numpy(x_hist),
            "y_future": torch.from_numpy(y_future),
        }

        if self.include_future_covariates:
            x_future = self.X_values[h_end:f_end]
            sample["x_future"] = torch.from_numpy(x_future)

        return sample


# ============================================================
# Split 1: Train / Val / Test temporale
# ============================================================

def temporal_train_val_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Split cronologico coerente (niente shuffle) tra train/val/test.
    Le serie devono avere stesso indice e lunghezza.

    train_ratio + val_ratio <= 1.0
    """
    if len(X) != len(y):
        raise ValueError("X e y devono avere stessa lunghezza per lo split temporale.")

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio e val_ratio non validi.")

    X_sorted = X.sort_index()
    y_sorted = y.sort_index()

    n = len(X_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X_sorted.iloc[:train_end], y_sorted.iloc[:train_end]
    X_val, y_val = X_sorted.iloc[train_end:val_end], y_sorted.iloc[train_end:val_end]
    X_test, y_test = X_sorted.iloc[val_end:], y_sorted.iloc[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# Split 2: Cross-Validation temporale (tipo TimeSeriesSplit)
# ============================================================

def temporal_cv_splits(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 5,
    test_size: Optional[int] = None,
) -> List[
    Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame, pd.DataFrame
    ]
]:
    """
    Genera split di Cross-Validation per serie temporali:
    - train = finestra che cresce progressivamente
    - val   = blocco subito successivo al train
    - test  = holdout finale fisso (uguale per tutti i fold)

    Simile a TimeSeriesSplit con un holdout finale.
    Ritorna una lista di tuple: (X_train, X_val, X_test, y_train, y_val, y_test).
    """

    if len(X) != len(y):
        raise ValueError("X e y devono avere stessa lunghezza per la CV temporale.")

    if n_splits < 2:
        raise ValueError("n_splits deve essere almeno 2.")

    X_sorted = X.sort_index()
    y_sorted = y.sort_index()
    n = len(X_sorted)

    if test_size is None:
        # (n_splits blocchi val) + 1 blocco test finale + train rimanente
        test_size = n // (n_splits + 2)
        if test_size == 0:
            raise ValueError("Serie troppo corta rispetto a n_splits/test_size.")

    test_start = n - test_size
    if test_start <= 0:
        raise ValueError("test_size troppo grande rispetto alla serie.")

    # blocco di test fisso, condiviso da tutti i fold
    X_test = X_sorted.iloc[test_start:]
    y_test = y_sorted.iloc[test_start:]

    X_tv = X_sorted.iloc[:test_start]
    y_tv = y_sorted.iloc[:test_start]
    n_tv = len(X_tv)

    splits: List[
        Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame,
            pd.DataFrame, pd.DataFrame, pd.DataFrame
        ]
    ] = []

    for k in range(n_splits):
        train_end = test_size * (k + 1)
        val_start = train_end
        val_end = val_start + test_size

        if val_end > n_tv:
            break

        X_train = X_tv.iloc[:train_end]
        y_train = y_tv.iloc[:train_end]
        X_val = X_tv.iloc[val_start:val_end]
        y_val = y_tv.iloc[val_start:val_end]

        splits.append((X_train, X_val, X_test, y_train, y_val, y_test))

    if not splits:
        raise ValueError("Nessuno split CV generato: controlla n_splits/test_size/len(X).")

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


# ============================================================
# Entry point: split + dataset + dataloader
# ============================================================

def prepare_dataloaders(
    X: pd.DataFrame,
    y: pd.DataFrame,
    mode: str = "train_val_test",   # "cv" per cross-validation
    config: Optional[PVDataConfig] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    n_splits: int = 5,
    test_size: Optional[int] = None,
    return_splits: bool = False,
):
    """
    Entry point unico per ottenere DataLoader (e opzionalmente gli split).

    mode = "train_val_test":
        ritorna train/val/test DataLoader; se return_splits=True restituisce anche gli split.

    mode = "cv":
        ritorna una lista di fold con i rispettivi DataLoader.
    """
    if config is None:
        config = PVDataConfig()

    if mode == "train_val_test":
        X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(
            X,
            y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        train_ds = PVForecastDataset(X_train, y_train, config)
        val_ds   = PVForecastDataset(X_val,   y_val,   config)
        test_ds  = PVForecastDataset(X_test,  y_test,  config)

        train_loader = build_dataloader(
            train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers
        )
        val_loader = build_dataloader(
            val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = build_dataloader(
            test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if return_splits:
            return (
                train_loader,
                val_loader,
                test_loader,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            )
        return train_loader, val_loader, test_loader

    if mode == "cv":
        splits = temporal_cv_splits(
            X,
            y,
            n_splits=n_splits,
            test_size=test_size,
        )

        folds: List[Dict[str, object]] = []
        for fold_idx, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(splits):
            train_ds = PVForecastDataset(X_train, y_train, config)
            val_ds   = PVForecastDataset(X_val,   y_val,   config)
            test_ds  = PVForecastDataset(X_test,  y_test,  config)

            train_loader = build_dataloader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            val_loader = build_dataloader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            test_loader = build_dataloader(
                test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

            folds.append(
                {
                    "fold": fold_idx,
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "test_loader": test_loader,
                }
            )

        return folds

    raise ValueError("mode deve essere 'train_val_test' oppure 'cv'")
