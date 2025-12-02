import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Tuple


def temporal_train_val_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split cronologico coerente (niente shuffle) tra train/val/test.
    Le serie devono avere stesso indice e lunghezza.
    """
    if len(X) != len(y):
        raise ValueError("X e y devono avere stessa lunghezza per lo split temporale.")

    X_sorted = X.sort_index()
    y_sorted = y.sort_index()

    n = len(X_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X_sorted.iloc[:train_end], y_sorted.iloc[:train_end]
    X_val, y_val = X_sorted.iloc[train_end:val_end], y_sorted.iloc[train_end:val_end]
    X_test, y_test = X_sorted.iloc[val_end:], y_sorted.iloc[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


@dataclass
class PVDataConfig:
    history_hours: int = 72
    horizon_hours: int = 24
    include_future_covariates: bool = False


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

        self.X = X.sort_index()
        self.y = y.sort_index()

        self.history = config.history_hours
        self.horizon = config.horizon_hours
        self.include_future_covars = config.include_future_covariates

        self.X_values = self.X.to_numpy(dtype=np.float32)
        self.y_values = self.y.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1)

        self.max_start = len(self.X_values) - (self.history + self.horizon) + 1
        if self.max_start <= 0:
            raise ValueError("Serie troppo corta rispetto a history+horizon.")

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.max_start:
            raise IndexError("Indice fuori range per il dataset.")

        h_start = idx
        h_end = idx + self.history
        f_end = h_end + self.horizon

        x_hist = self.X_values[h_start:h_end]
        y_future = self.y_values[h_end:f_end]

        sample = {
            "x_hist": torch.from_numpy(x_hist),
            "y_future": torch.from_numpy(y_future),
        }

        if self.include_future_covars:
            x_future = self.X_values[h_end:f_end]
            sample["x_future"] = torch.from_numpy(x_future)

        return sample


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
