from dataclasses import dataclass, field
import os
import pickle
from typing import Any, Dict, List, Optional
import numpy as np

from .splits import select_open_set_classes
from .transforms import preprocess_signals


@dataclass
class OSRDataBundle:
    """Standardized open-set dataset container for SEI and future signal recognition tasks."""

    train_known: Dict[str, np.ndarray]
    val_known: Dict[str, np.ndarray]
    eval_known: Dict[str, np.ndarray]
    eval_unknown: Dict[str, np.ndarray]
    known_classes: List[str]
    unknown_classes: List[str]
    signal_length: int
    task_type: str = "sei_osr"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_known(self) -> int:
        return len(self.known_classes)

    @property
    def num_unknown(self) -> int:
        return len(self.unknown_classes)


def _flatten_eval_dict(eval_section: Dict[str, Any], train_keys: List[str]) -> Dict[str, np.ndarray]:
    """Normalize evaluation section whether keyed by class name or nested under {'known', 'unknown'}."""
    if "known" in eval_section and "unknown" in eval_section and isinstance(eval_section["known"], dict):
        merged: Dict[str, np.ndarray] = {}
        for k, v in eval_section["known"].items():
            merged[k] = np.asarray(v)
        for k, v in eval_section["unknown"].items():
            uk_name = k if k not in merged else f"Unknown_{k}"
            merged[uk_name] = np.asarray(v)
        return merged
    return {k: np.asarray(v) for k, v in eval_section.items()}


def _slice_or_resplit_class(
    train_arr: np.ndarray,
    val_arr: np.ndarray,
    eval_arr: np.ndarray,
    train_samples: Optional[int],
    val_samples: Optional[int],
    eval_samples: Optional[int],
    pool_and_resplit: bool,
    rng: np.random.RandomState,
):
    """
    Slice class arrays to requested sample counts.
    If pool_and_resplit is True (or train_samples > len(train_arr)), pool train+val+eval and split.
    """
    need_pool = pool_and_resplit or (train_samples is not None and train_samples > len(train_arr))
    if need_pool:
        pooled = np.concatenate([train_arr, val_arr, eval_arr], axis=0)
        idx = rng.permutation(len(pooled))
        pooled = pooled[idx]
        n_total = len(pooled)
        n_tr = train_samples if train_samples is not None else len(train_arr)
        rem = n_total - n_tr
        if rem <= 0:
            raise ValueError(f"Requested train_samples_per_class={n_tr} leaves no samples for val/eval (total={n_total}).")
        n_val = val_samples if val_samples is not None else min(len(val_arr), max(1, rem // 3))
        n_ev = eval_samples if eval_samples is not None else (rem - n_val)
        if n_tr + n_val + n_ev > n_total:
            n_ev = max(1, n_total - n_tr - n_val)
        return (
            pooled[:n_tr],
            pooled[n_tr : n_tr + n_val],
            pooled[n_tr + n_val : n_tr + n_val + n_ev],
        )

    tr_out = train_arr[:train_samples] if train_samples is not None else train_arr
    val_out = val_arr[:val_samples] if val_samples is not None else val_arr
    ev_out = eval_arr[:eval_samples] if eval_samples is not None else eval_arr
    return tr_out, val_out, ev_out


def load_osr_dataset(data_cfg: Dict[str, Any], default_seed: int = 42) -> OSRDataBundle:
    """
    Load an open-set pickle dataset and apply class splitting, sample-count slicing, and preprocessing.

    Supported data_cfg keys:
      - path (str): Path to .pickle file
      - split_mode (str): 'fixed' | 'random' | 'explicit' (default: 'fixed')
      - split_seed (int): Seed for class partition & sample resplitting (default: default_seed)
      - num_known (int, optional): Number of known classes
      - num_unknown (int, optional): Number of unknown classes
      - known_classes (list[str], optional): Explicit list of known class names
      - unknown_classes (list[str], optional): Explicit list of unknown class names
      - train_samples_per_class (int, optional): e.g. 10 (few-shot), 100 (standard), 500/1000/3000 (full)
      - val_samples_per_class (int, optional): Validation samples per known class
      - eval_samples_per_class (int, optional): Evaluation samples per class (e.g. 500 on WIDEFT)
      - pool_and_resplit (bool): Pool train+val+eval before slicing (default: False)
      - remove_dc (bool): Subtract per-frame DC offset (default: False)
      - compensate_cfo (bool): Coarse CFO compensation (default: False)
      - normalize_rms (bool): Normalize per-frame RMS power to 1 (default: False)
    """
    path = data_cfg.get("path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset pickle not found at: {path}")

    with open(path, "rb") as f:
        raw = pickle.load(f)

    train_sec: Dict[str, np.ndarray] = {k: np.asarray(v) for k, v in raw["training"].items()}
    val_sec: Dict[str, np.ndarray] = {k: np.asarray(v) for k, v in raw["validation"].items()}
    eval_sec: Dict[str, np.ndarray] = _flatten_eval_dict(raw["evaluation"], list(train_sec.keys()))

    split_seed = int(data_cfg.get("split_seed", default_seed))
    split_mode = str(data_cfg.get("split_mode", "fixed"))

    known_classes, unknown_classes = select_open_set_classes(
        train_keys=list(train_sec.keys()),
        eval_keys=list(eval_sec.keys()),
        split_mode=split_mode,
        num_known=data_cfg.get("num_known"),
        num_unknown=data_cfg.get("num_unknown"),
        known_classes=data_cfg.get("known_classes"),
        unknown_classes=data_cfg.get("unknown_classes"),
        split_seed=split_seed,
    )

    train_n = data_cfg.get("train_samples_per_class")
    val_n = data_cfg.get("val_samples_per_class")
    eval_n = data_cfg.get("eval_samples_per_class")
    pool_and_resplit = bool(data_cfg.get("pool_and_resplit", False))

    remove_dc = bool(data_cfg.get("remove_dc", False))
    compensate_cfo = bool(data_cfg.get("compensate_cfo", False))
    normalize_rms = bool(data_cfg.get("normalize_rms", False))

    rng = np.random.RandomState(split_seed)
    train_known: Dict[str, np.ndarray] = {}
    val_known: Dict[str, np.ndarray] = {}
    eval_known: Dict[str, np.ndarray] = {}
    eval_unknown: Dict[str, np.ndarray] = {}

    for k in known_classes:
        tr_a, val_a, ev_a = _slice_or_resplit_class(
            train_sec[k],
            val_sec[k],
            eval_sec[k],
            train_samples=train_n,
            val_samples=val_n,
            eval_samples=eval_n,
            pool_and_resplit=pool_and_resplit,
            rng=rng,
        )
        train_known[k] = preprocess_signals(tr_a, remove_dc, compensate_cfo, normalize_rms)
        val_known[k] = preprocess_signals(val_a, remove_dc, compensate_cfo, normalize_rms)
        eval_known[k] = preprocess_signals(ev_a, remove_dc, compensate_cfo, normalize_rms)

    for u in unknown_classes:
        ev_u = eval_sec[u]
        if eval_n is not None:
            ev_u = ev_u[:eval_n]
        eval_unknown[u] = preprocess_signals(ev_u, remove_dc, compensate_cfo, normalize_rms)

    first_arr = train_known[known_classes[0]]
    signal_length = int(first_arr.shape[1])

    return OSRDataBundle(
        train_known=train_known,
        val_known=val_known,
        eval_known=eval_known,
        eval_unknown=eval_unknown,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        signal_length=signal_length,
        task_type=str(data_cfg.get("task_type", "sei_osr")),
        metadata={
            "source_path": path,
            "split_mode": split_mode,
            "split_seed": split_seed,
            "train_samples_per_class": int(first_arr.shape[0]),
            "total_train_samples": int(sum(len(v) for v in train_known.values())),
            "total_val_samples": int(sum(len(v) for v in val_known.values())),
            "total_eval_known_samples": int(sum(len(v) for v in eval_known.values())),
            "total_eval_unknown_samples": int(sum(len(v) for v in eval_unknown.values())),
        },
    )
