from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple
import copy
import math
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.training import fit

SearchSpace = Mapping[str, Any]


@dataclass
class TrialResult:
    '''Funzione di utilità per memorizzare i risultati di un singolo trial di random search.
    '''
    params: Dict[str, Any]
    score: float
    metrics: Dict[str, float]
    model_key: str


@dataclass
class RandomSearchResult:
    '''Funzione di utilità per memorizzare i risultati complessivi della random search.
    '''
    best_params: Dict[str, Any]
    best_score: float
    best_metrics: Dict[str, float]
    trials: List[TrialResult]


def _get_model_key(cfg: Any, model_builder: Optional[Callable[..., nn.Module]] = None) -> str:
    '''
    Estrae la chiave del modello dalla configurazione o dal costruttore del modello.
    '''
    model_cfg = getattr(cfg, "model", None)
    for attr in ("model_type", "model_name", "name", "kind", "arch"):
        if model_cfg is not None and hasattr(model_cfg, attr):
            value = getattr(model_cfg, attr)
            if value:
                return str(value)
    if model_builder is not None and hasattr(model_builder, "__name__"):
        return str(model_builder.__name__)
    return "default"


def _resolve_search_space(cfg: Any, model_key: str, override: Optional[SearchSpace]) -> SearchSpace:
    '''
    Risolve lo spazio di ricerca per un dato modello, tenendo conto di eventuali sovrascritture.
    '''
    if override is not None:
        if isinstance(override, Mapping):
            if model_key in override:
                return override[model_key]
            if "default" in override:
                return override["default"]
        return override

    candidates = []
    for attr in ("hyperparam_search", "hparam_search", "search_space", "random_search"):
        if hasattr(cfg, attr):
            candidates.append(getattr(cfg, attr))

    model_cfg = getattr(cfg, "model", None)
    if model_cfg is not None:
        for attr in ("hyperparam_search", "hparam_search", "search_space"):
            if hasattr(model_cfg, attr):
                candidates.append(getattr(model_cfg, attr))

    training_cfg = getattr(cfg, "training", None)
    if training_cfg is not None:
        for attr in ("hyperparam_search", "hparam_search", "search_space"):
            if hasattr(training_cfg, attr):
                candidates.append(getattr(training_cfg, attr))

    for space in candidates:
        if isinstance(space, Mapping):
            if model_key in space:
                return space[model_key]
            if "default" in space:
                return space["default"]
            return space

    raise ValueError(
        "Search space not found. Provide search_space or set cfg.hyperparam_search."
    )


def _set_by_path(obj: Any, path: str, value: Any) -> None:
    '''
    Imposta un valore in un oggetto annidato utilizzando un percorso specificato.
    '''
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        if isinstance(target, Mapping):
            target = target[part]
        else:
            target = getattr(target, part)

    last = parts[-1]
    if isinstance(target, Mapping):
        target[last] = value
    else:
        setattr(target, last, value)


def _sample_choice(rng: random.Random, values: Iterable[Any]) -> Any:
    '''
    Campiona un valore casuale da un insieme di valori.
    '''
    values_list = list(values)
    if not values_list:
        raise ValueError("Choice space is empty.")
    return values_list[rng.randrange(len(values_list))]


def _sample_from_spec(rng: random.Random, spec: Any) -> Any:
    '''
    Campiona un valore casuale da una specifica.
    '''
    if isinstance(spec, Mapping):
        spec_type = str(spec.get("type", "choice"))
        if spec_type == "choice":
            return _sample_choice(rng, spec.get("values", []))
        if spec_type == "bool":
            p = float(spec.get("p", 0.5))
            return rng.random() < p
        if spec_type == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            if step <= 0:
                raise ValueError("int step must be positive.")
            values = list(range(low, high + 1, step))
            return _sample_choice(rng, values)
        if spec_type in ("float", "logfloat"):
            low = float(spec["low"])
            high = float(spec["high"])
            if spec_type == "logfloat":
                if low <= 0 or high <= 0:
                    raise ValueError("logfloat range must be > 0.")
                return math.exp(rng.uniform(math.log(low), math.log(high)))
            return rng.uniform(low, high)
        raise ValueError(f"Unsupported spec type: {spec_type}")

    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec):
            low, high = spec
            if isinstance(low, int) and isinstance(high, int):
                return rng.randint(low, high)
            return rng.uniform(float(low), float(high))
        return _sample_choice(rng, spec)

    return spec


def _sample_params(rng: random.Random, space: SearchSpace) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key, spec in space.items():
        params[key] = _sample_from_spec(rng, spec)
    return params


def _sync_model_dims_from_loader(cfg: Any, train_loader: DataLoader) -> None:
    dataset = getattr(train_loader, "dataset", None)
    if dataset is None:
        return

    if hasattr(dataset, "X_values"):
        base_size = int(getattr(dataset, "X_values").shape[1])
        include_past_target = bool(
            getattr(dataset, "include_past_target", False)
            or getattr(getattr(dataset, "config", None), "include_past_target", False)
        )
        if include_past_target:
            base_size += 1
        if hasattr(cfg, "model"):
            setattr(cfg.model, "input_size", base_size)

    if hasattr(dataset, "horizon") and hasattr(cfg, "model"):
        setattr(cfg.model, "horizon", int(getattr(dataset, "horizon")))


def _set_global_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _reset_loader_seed(loader: Optional[DataLoader], seed: int) -> None:
    if loader is None:
        return
    generator = getattr(loader, "generator", None)
    if generator is not None:
        try:
            generator.manual_seed(int(seed))
        except Exception:
            pass


def _pick_metric_from_fit_result(
    fit_result: Any,
    metric: str,
    mode: str,
) -> Tuple[float, Dict[str, float]]:
    metric = metric.lower()
    mode = mode.lower()

    val_losses = list(getattr(fit_result, "val_losses", []))
    val_rmse = list(getattr(fit_result, "val_rmse", []))
    val_mase = list(getattr(fit_result, "val_mase", []))
    train_losses = list(getattr(fit_result, "train_losses", []))

    if val_losses:
        if metric == "loss":
            best_idx = int(np.argmin(val_losses))
        elif metric == "rmse":
            best_idx = int(np.argmin(val_rmse)) if val_rmse else int(np.argmin(val_losses))
        elif metric == "mase":
            best_idx = int(np.argmin(val_mase)) if val_mase else int(np.argmin(val_losses))
        else:
            raise ValueError("metric must be one of: loss, rmse, mase")

        metrics = {
            "loss": float(val_losses[best_idx]),
            "rmse": float(val_rmse[best_idx]) if val_rmse else float("nan"),
            "mase": float(val_mase[best_idx]) if val_mase else float("nan"),
        }
    else:
        if not train_losses:
            raise ValueError("No losses found in fit result.")
        metrics = {"loss": float(train_losses[-1]), "rmse": float("nan"), "mase": float("nan")}

    score = metrics[metric]
    if mode == "max":
        score = -score
    return score, metrics


def random_search_cv(
    model_builder: Callable[[Any], nn.Module],
    fold_loaders: List[Mapping[str, Any]],
    cfg: Any,
    *,
    search_space: Optional[SearchSpace] = None,
    n_trials: int = 20,
    metric: str = "loss",
    mode: str = "min",
    seed: int = 42,
    deterministic: Optional[bool] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> RandomSearchResult:
    """
    Esegue una ricerca casuale con cross-validation sui parametri del modello.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")
    if not fold_loaders:
        raise ValueError("fold_loaders must be non-empty")

    device = device or torch.device("cpu")
    rng = random.Random(seed)
    if deterministic is None:
        deterministic = bool(getattr(getattr(cfg, "training", None), "deterministic", True))

    model_key = _get_model_key(cfg, model_builder=model_builder)
    space = _resolve_search_space(cfg, model_key, search_space)

    best_params: Dict[str, Any] = {}
    best_score = float("inf")
    best_metrics: Dict[str, float] = {}
    trials: List[TrialResult] = []

    for trial_idx in range(1, n_trials + 1):
        params = _sample_params(rng, space)
        fold_scores: List[float] = []
        fold_metrics: List[Dict[str, float]] = []

        for fold in fold_loaders:
            cfg_trial = copy.deepcopy(cfg)
            for key, value in params.items():
                _set_by_path(cfg_trial, key, value)

            train_loader = fold.get("train_loader")
            val_loader = fold.get("val_loader")

            if train_loader is None:
                raise ValueError("Each fold must provide a train_loader.")

            _sync_model_dims_from_loader(cfg_trial, train_loader)

            fold_id = int(fold.get("fold", 0))
            trial_seed = int(seed) + trial_idx * 1000 + fold_id * 10
            _set_global_seed(trial_seed, deterministic)
            _reset_loader_seed(train_loader, trial_seed)
            _reset_loader_seed(val_loader, trial_seed + 1)

            model = model_builder(cfg_trial.model)

            mase_m = getattr(getattr(cfg_trial, "baseline", None), "mase_seasonal_m", 24)
            y_train_insample = getattr(getattr(train_loader, "dataset", None), "y_values", None)

            fit_result = fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=int(getattr(cfg_trial.training, "epochs", 10)),
                lr=float(getattr(cfg_trial.training, "lr", 1e-3)),
                device=device,
                y_train_insample=y_train_insample,
                mase_m=int(mase_m),
                keep_best_on_val=True,
                best_metric=metric,
                early_stopping=bool(getattr(cfg_trial.training, "early_stopping", False)),
                patience=int(getattr(cfg_trial.training, "patience", 5)),
                min_delta=float(getattr(cfg_trial.training, "min_delta", 0.0)),
                verbose=verbose,
            )

            score, metrics = _pick_metric_from_fit_result(fit_result, metric, mode)
            fold_scores.append(score)
            fold_metrics.append(metrics)

        avg_score = float(np.mean(fold_scores))
        avg_metrics = {
            "loss": float(np.nanmean([m.get("loss", float("nan")) for m in fold_metrics])),
            "rmse": float(np.nanmean([m.get("rmse", float("nan")) for m in fold_metrics])),
            "mase": float(np.nanmean([m.get("mase", float("nan")) for m in fold_metrics])),
        }

        trials.append(
            TrialResult(
                params=params,
                score=avg_score,
                metrics=avg_metrics,
                model_key=model_key,
            )
        )

        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            best_metrics = avg_metrics

        if verbose:
            print(
                f"[RANDOM_SEARCH_CV][{trial_idx}/{n_trials}] params={params} "
                f"| metrics={avg_metrics}"
            )

    return RandomSearchResult(
        best_params=best_params,
        best_score=best_score,
        best_metrics=best_metrics,
        trials=trials,
    )
