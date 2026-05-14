from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"YAML config is empty: {path}")

    if not isinstance(data, dict):
        raise TypeError(f"YAML config must load to dict, got {type(data)} from {path}")

    return data


def write_yaml(data: dict[str, Any], path: str | Path) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def deep_merge_dicts(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:

    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def deep_get(
    data: dict[str, Any],
    path: str,
    default: Any = None,
    sep: str = ".",
) -> Any:

    current: Any = data

    for part in path.split(sep):
        if not isinstance(current, dict):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


def load_config(
    config_path: str | Path,
    include_linked_configs: bool = True,
) -> dict[str, Any]:

    config_path = Path(config_path)
    cfg = load_yaml(config_path)

    cfg["_config_path"] = str(config_path)

    if not include_linked_configs:
        return cfg

    root = config_path.parent.parent.parent if "configs" in config_path.parts else Path.cwd()

    resolved: dict[str, Any] = {}

    experiment = cfg.get("experiment", {})

    linked_keys = {
        "dataset_config": "dataset",
        "teacher_config": "teacher",
        "student_config": "student",
        "kd_config": "kd",
        "cache_config": "cache",
    }

    for key, target_name in linked_keys.items():
        linked_path_value = experiment.get(key)

        if linked_path_value is None:
            continue

        linked_path = Path(linked_path_value)

        if not linked_path.is_absolute():
            linked_path = root / linked_path

        if not linked_path.exists():
            # Fallback: relative to current working directory.
            linked_path = Path(linked_path_value)

        linked_cfg = load_yaml(linked_path)
        linked_cfg["_config_path"] = str(linked_path)
        resolved[target_name] = linked_cfg

    cfg["resolved"] = resolved
    return cfg


def save_config_snapshot(
    config: dict[str, Any],
    output_dir: str | Path,
    name: str = "resolved_config.json",
) -> Path:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / name

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return out_path


def copy_config_file(
    source: str | Path,
    output_dir: str | Path,
) -> Path:

    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Config file not found: {source}")

    dest = output_dir / source.name
    shutil.copy2(source, dest)

    return dest