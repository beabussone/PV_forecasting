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
    """
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

        # ordinamento temporale
        self.X = X.sort_index()
        self.y = y.sort_index()

        self.history = config.history_hours
        self.horizon = config.horizon_hours
        self.include_future_covars = config.include_future_covariates

        # array numpy
        self.X_values = self.X.to_numpy(dtype=np.float32)
        # assumiamo che la prima colonna di y sia la label kwp
        self.y_values = self.y.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1)

        # numero massimo di indici di partenza validi
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
# High-level: Train/Val/Test -> Dataset + DataLoader
# ============================================================

def build_pv_datasets_train_val_test(
    X: pd.DataFrame,
    y: pd.DataFrame,
    config: Optional[PVDataConfig] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Wrapper che:
    1) fa lo split temporale train/val/test
    2) costruisce 3 PVForecastDataset
    """
    if config is None:
        config = PVDataConfig()

    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(
        X, y, train_ratio=train_ratio, val_ratio=val_ratio
    )

    train_ds = PVForecastDataset(X_train, y_train, config)
    val_ds = PVForecastDataset(X_val, y_val, config)
    test_ds = PVForecastDataset(X_test, y_test, config)

    return train_ds, val_ds, test_ds


def build_pv_dataloaders_train_val_test(
    X: pd.DataFrame,
    y: pd.DataFrame,
    config: Optional[PVDataConfig] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 0,
):
    """
    Wrapper ancora più high-level:
    ritorna direttamente i 3 DataLoader:
    - train_loader, val_loader, test_loader
    """
    train_ds, val_ds, test_ds = build_pv_datasets_train_val_test(
        X, y, config=config, train_ratio=train_ratio, val_ratio=val_ratio
    )

    train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = build_dataloader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# ============================================================
# High-level: CV temporale -> Dataset + DataLoader per fold
# ============================================================

def build_pv_cv_datasets(
    X: pd.DataFrame,
    y: pd.DataFrame,
    config: Optional[PVDataConfig] = None,
    n_splits: int = 5,
    test_size: Optional[int] = None,
) -> List[Dict[str, object]]:
    """
    Costruisce i dataset per ogni fold della CV temporale.
    Ritorna una lista di dict, uno per fold:
    {
        "fold": idx_fold,
        "train": PVForecastDataset(...),
        "val":   PVForecastDataset(...),
        "test":  PVForecastDataset(...),
    }
    """
    if config is None:
        config = PVDataConfig()

    splits = temporal_cv_splits(X, y, n_splits=n_splits, test_size=test_size)
    folds: List[Dict[str, object]] = []

    for fold_idx, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(splits):
        train_ds = PVForecastDataset(X_train, y_train, config)
        val_ds = PVForecastDataset(X_val, y_val, config)
        test_ds = PVForecastDataset(X_test, y_test, config)
        folds.append(
            {
                "fold": fold_idx,
                "train": train_ds,
                "val": val_ds,
                "test": test_ds,
            }
        )

    return folds


def build_pv_cv_dataloaders(
    X: pd.DataFrame,
    y: pd.DataFrame,
    config: Optional[PVDataConfig] = None,
    n_splits: int = 5,
    test_size: Optional[int] = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> List[Dict[str, object]]:
    """
    Come build_pv_cv_datasets ma ritorna direttamente i DataLoader per ogni fold:
    [
        {
            "fold": k,
            "train_loader": ...,
            "val_loader":   ...
            "test_loader":  ...
        },
        ...
    ]
    """
    folds_ds = build_pv_cv_datasets(
        X, y, config=config, n_splits=n_splits, test_size=test_size
    )

    folds_loaders: List[Dict[str, object]] = []
    for f in folds_ds:
        train_loader = build_dataloader(
            f["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = build_dataloader(
            f["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = build_dataloader(
            f["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        folds_loaders.append(
            {
                "fold": f["fold"],
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
            }
        )

    return folds_loaders


# ============================================================
# Nuova organizzazione: split e dataloader in due step
# ============================================================

def prepare_data_splits(
    X: pd.DataFrame,
    y: pd.DataFrame,
    mode: str = "train_val_test",   # "cv" per cross-validation
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    n_splits: int = 5,
    test_size: Optional[int] = None,
):
    """
    Genera solo gli split, da invocare prima di OHE / feature engineering.
    Tenere separato lo split evita leakage quando si fittano encoder/scaler.

    mode = "train_val_test"
        ritorna (X_train, X_val, X_test, y_train, y_val, y_test)

    mode = "cv"
        ritorna una lista di tuple (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if mode == "train_val_test":
        return temporal_train_val_test_split(
            X,
            y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
    elif mode == "cv":
        return temporal_cv_splits(
            X,
            y,
            n_splits=n_splits,
            test_size=test_size,
        )
    else:
        raise ValueError("mode deve essere 'train_val_test' oppure 'cv'")


def build_dataloaders_from_splits(
    splits,
    mode: str = "train_val_test",
    config: Optional[PVDataConfig] = None,
    batch_size: int = 64,
    num_workers: int = 0,
):
    """
    Costruisce i DataLoader partendo da split gia pronti
    (quindi dopo OHE/feature engineering/scaling).
    """
    if config is None:
        config = PVDataConfig()

    if mode == "train_val_test":
        X_train, X_val, X_test, y_train, y_val, y_test = splits

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

        return train_loader, val_loader, test_loader

    elif mode == "cv":
        folds_loaders: List[Dict[str, object]] = []
        for fold_idx, split in enumerate(splits):
            X_train, X_val, X_test, y_train, y_val, y_test = split

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

            folds_loaders.append(
                {
                    "fold": fold_idx,
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "test_loader": test_loader,
                }
            )

        if not folds_loaders:
            raise ValueError("Nessun fold generato: controlla gli split in input.")

        return folds_loaders

    else:
        raise ValueError("mode deve essere 'train_val_test' oppure 'cv'")


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
    return_splits: bool = False,    # <<< NOVITÀ
):
    """
    Entry point unico.

    mode = "train_val_test"
        - se return_splits=False:
              ritorna (train_loader, val_loader, test_loader)
        - se return_splits=True:
              ritorna (train_loader, val_loader, test_loader,
                       X_train, X_val, X_test,
                       y_train, y_val, y_test)

    mode = "cv"
        - ritorna una lista di fold:
              [ {"fold": k, "train_loader":..., "val_loader":..., "test_loader":...}, ... ]
          (qui return_splits per ora viene ignorato)
    """
    if config is None:
        config = PVDataConfig()

    if mode == "train_val_test":
        # 1) split temporale
        X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(
            X,
            y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        # 2) dataset
        train_ds = PVForecastDataset(X_train, y_train, config)
        val_ds   = PVForecastDataset(X_val,   y_val,   config)
        test_ds  = PVForecastDataset(X_test,  y_test,  config)

        # 3) dataloader
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
        else:
            return train_loader, val_loader, test_loader

    elif mode == "cv":
        # per la CV lasciamo la semantica invariata
        return build_pv_cv_dataloaders(
            X, y,
            config=config,
            n_splits=n_splits,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    else:
        raise ValueError("mode deve essere 'train_val_test' oppure 'cv'")
