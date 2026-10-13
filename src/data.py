import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_split(data_dir, split):
    if split not in {"train", "val", "test"}:
        raise ValueError(split)
    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("image_id trùng trong split.")
    return rows


def subset_train(rows, fraction, seed):
    rows = sorted(rows, key=lambda r: r["image_id"])
    if fraction == 1:
        return rows
    if not 0 < fraction <= 1:
        raise ValueError(fraction)
    return random.Random(seed).sample(rows, max(1, int(len(rows) * fraction)))


def load_image(row, data_dir):
    from PIL import Image
    with Image.open(Path(data_dir) / row["image_path"]) as image:
        image = image.convert("RGB")
        if image.size != (row["width"], row["height"]):
            raise ValueError("Kích thước ảnh không khớp split.")
        return image


def inspect_data(data_dir):
    meta = json.loads((Path(data_dir) / "splits/meta.json").read_text(encoding="utf-8"))
    groups, ids, summary = {}, {}, {}
    for split in ("train", "val", "test"):
        rows = read_split(data_dir, split)
        groups[split] = {r["session"] for r in rows}
        ids[split] = {r["image_id"] for r in rows}
        counts = Counter(o["label"] for r in rows for o in r["objects"])
        summary[split] = {"images": len(rows), "boxes": dict(counts)}
        expected = meta["counts"][split]
        assert len(rows) == expected["images"]
        assert all(counts[c] == n for c, n in expected["boxes"].items())
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (groups[a] & groups[b]), "Rò rỉ session."
        assert not (ids[a] & ids[b]), "Rò rỉ image_id."
    print(json.dumps({"dataset_version": meta["dataset_version"], "counts": summary}, indent=2))
    return meta
