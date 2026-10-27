"""Small shared helpers: jsonl I/O, seeding, config loading, timing."""
import json
import os
import random
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]], mode: str = "w") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    write_jsonl(path, rows, mode="a")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """Parse CLI overrides of the form key=value (value parsed as YAML)."""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"override must be key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def load_config(path: str, overrides: List[str] = ()) -> Dict[str, Any]:
    """Load a YAML config. A config may name a `base:` config that it extends."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("base"))
        base = load_config(base_path)
        base.update(cfg)
        cfg = base
    cfg.update(parse_overrides(list(overrides)))
    return cfg


def short_model_name(name: str) -> str:
    """'Qwen/Qwen3-1.7B' -> 'qwen3-1.7b'."""
    return name.rstrip("/").split("/")[-1].lower()


@contextmanager
def timer(store: Dict[str, float], key: str):
    t0 = time.time()
    yield
    store[key] = store.get(key, 0.0) + time.time() - t0
