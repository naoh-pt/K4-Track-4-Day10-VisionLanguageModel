import copy
import hashlib
import json
from pathlib import Path

import yaml


def load_config(path, fraction=1.0, batch_size=None, grad_accum=None):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if cfg["name"] not in {"E0", "E1", "E2", "E3"}:
        raise ValueError("E4 dùng cấu hình đối chứng E1–E3 và --fraction; không train e4.yaml trực tiếp.")
    cfg.setdefault("max_new_tokens", 256)
    cfg.setdefault("seed", 42)
    cfg["train_fraction"] = fraction
    if not 0 < fraction <= 1:
        raise ValueError("fraction phải thuộc (0,1].")
    if batch_size is not None or grad_accum is not None:
        if (batch_size, grad_accum) != (1, 8):
            raise ValueError("Chỉ hỗ trợ phương án OOM GUIDE: --batch-size 1 --grad-accum 8.")
        cfg.update(batch_size=1, grad_accum=8)
    return cfg


def fingerprint(cfg, dataset_version):
    payload = {"config": copy.deepcopy(cfg), "dataset_version": dataset_version}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
