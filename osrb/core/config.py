import copy
import json
import os
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json") or yaml is None:
            return json.load(f)
        data = yaml.safe_load(f)
        return data if data is not None else {}


def save_yaml(data: Dict[str, Any], path: str) -> None:
    """Save configuration dictionary to YAML (or JSON if PyYAML is unavailable)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if path.endswith(".json") or yaml is None:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def apply_dot_overrides(cfg: Dict[str, Any], overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """Apply CLI key=value dot-notation overrides (e.g. 'data.train_samples_per_class=100')."""
    if not overrides:
        return cfg
    result = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected format: key.subkey=value")
        key_path, raw_val = item.split("=", 1)
        if yaml is not None:
            val = yaml.safe_load(raw_val)
        else:
            try:
                val = json.loads(raw_val)
            except Exception:
                val = raw_val
        keys = key_path.strip().split(".")
        curr = result
        for k in keys[:-1]:
            if k not in curr or not isinstance(curr[k], dict):
                curr[k] = {}
            curr = curr[k]
        curr[keys[-1]] = val
    return result


def resolve_config(
    config_path: Optional[str] = None,
    dataset_config_path: Optional[str] = None,
    method_config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Resolve a complete experiment config from base files, includes, and CLI overrides."""
    cfg: Dict[str, Any] = {
        "task": {"type": "sei_osr"},
        "seed": 42,
        "data": {},
        "augment": {},
        "model": {},
        "method": {},
        "train": {},
        "calib": {},
        "eval": {},
    }

    if dataset_config_path:
        ds_cfg = load_yaml(dataset_config_path)
        cfg = merge_configs(cfg, ds_cfg if "data" in ds_cfg else {"data": ds_cfg})

    if method_config_path:
        m_cfg = load_yaml(method_config_path)
        cfg = merge_configs(cfg, m_cfg if "method" in m_cfg else {"method": m_cfg})

    if config_path:
        main_cfg = load_yaml(config_path)
        base_dir = os.path.dirname(os.path.abspath(config_path))
        # Support optional 'includes' list or 'dataset_config' / 'method_config' keys
        if "dataset_config" in main_cfg:
            rel_ds = main_cfg.pop("dataset_config")
            p_ds = rel_ds if os.path.isabs(rel_ds) else os.path.join(base_dir, rel_ds)
            if not os.path.exists(p_ds):
                p_ds = rel_ds
            ds_cfg = load_yaml(p_ds)
            cfg = merge_configs(cfg, ds_cfg if "data" in ds_cfg else {"data": ds_cfg})
        if "method_config" in main_cfg:
            rel_m = main_cfg.pop("method_config")
            p_m = rel_m if os.path.isabs(rel_m) else os.path.join(base_dir, rel_m)
            if not os.path.exists(p_m):
                p_m = rel_m
            m_cfg = load_yaml(p_m)
            cfg = merge_configs(cfg, m_cfg if "method" in m_cfg else {"method": m_cfg})
        cfg = merge_configs(cfg, main_cfg)

    cfg = apply_dot_overrides(cfg, overrides)
    return cfg
