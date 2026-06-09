"""从 YAML 加载超参（公开版不内置论文调优默认值）。"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_hyperparams(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"超参配置文件不存在: {path}\n"
            "请复制 hyperparams.example.yaml 为 hyperparams.yaml 并自行标定。"
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for section in ("training", "heuristic", "inference"):
        if section not in cfg:
            raise KeyError(f"配置缺少节 [{section}]，请参考 hyperparams.example.yaml")
    return cfg


def default_example_config_path() -> str:
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(root, "hyperparams.example.yaml")
