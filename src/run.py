"""CLI Colab: python -m src.run --help. Không import model khi chỉ xem help."""
import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def main():
    # PowerShell/PTY Windows có thể mặc định cp1252 thay vì UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--data", default="data")
    train.add_argument("--out", required=True)
    train.add_argument("--fraction", type=float, default=1.0)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--grad-accum", type=int)
    train.add_argument("--amp-init-scale", type=float)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--smoke-steps", type=int, default=2)
    train.add_argument("--smoke-dir")
    train.add_argument("--backup")
    infer = commands.add_parser("predict")
    infer.add_argument("--config", default="configs/e0.yaml")
    infer.add_argument("--checkpoint")
    infer.add_argument("--data", default="data")
    infer.add_argument("--split", choices=["val", "test"], default="val")
    infer.add_argument("--limit", type=int)
    infer.add_argument("--batch-size", type=int, default=4)
    infer.add_argument("--out", required=True)
    score = commands.add_parser("score")
    score.add_argument("--pred", required=True)
    score.add_argument("--data", default="data")
    score.add_argument("--split", choices=["val", "test"], default="val")
    score.add_argument("--name", required=True)
    score.add_argument("--checkpoint")
    score.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "train":
        from .train import train as train_run
        cfg = load_config(args.config, args.fraction, args.batch_size, args.grad_accum)
        if args.amp_init_scale is not None:
            import math
            if not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0:
                parser.error("amp-init-scale phải hữu hạn và dương.")
            cfg["amp_init_scale"] = args.amp_init_scale
        train_run(cfg, args.data, args.out, args.smoke, args.smoke_steps, args.backup, args.smoke_dir)
    elif args.command == "predict":
        from .data import read_split
        from .modeling import load_base
        from .inference import predict
        rows = read_split(args.data, args.split)
        if args.limit is not None:
            if args.split != "val" or args.limit < 1:
                parser.error("--limit chỉ dùng với val và phải dương; test luôn chạy đủ.")
            rows = rows[:args.limit]
        if args.batch_size < 1:
            parser.error("batch-size phải dương.")
        if args.checkpoint:
            from .checkpoints import load_checkpoint
            checkpoint = Path(args.checkpoint)
            cfg = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
            run = json.loads((checkpoint / "run.json").read_text(encoding="utf-8"))
            if run["smoke"]:
                parser.error("Checkpoint smoke không được dùng làm kết quả chính thức.")
            version = json.loads((Path(args.data) / "splits/meta.json").read_text(encoding="utf-8"))["dataset_version"]
            if version != run["dataset_version"]:
                parser.error("Dataset checkpoint khác dataset hiện tại.")
        else:
            cfg = load_config(args.config)
            if cfg["name"] != "E0":
                parser.error("E1–E4 cần --checkpoint.")
        model, processor = load_base(cfg)
        if args.checkpoint:
            model = load_checkpoint(model, args.checkpoint)
        predict(model, processor, rows, args.data, args.out, args.batch_size, cfg["max_new_tokens"])
    else:
        # Gọi nguyên trọng tài, không tự tính lại evidence.
        from scripts.score import main as score_main
        from .data import read_jsonl, read_split
        if Path(args.out).exists():
            raise FileExistsError(args.out)
        expected = {r["image_id"] for r in read_split(args.data, args.split)}
        predictions = read_jsonl(args.pred)
        actual = [r["image_id"] for r in predictions]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("Chỉ chấm evidence khi có đúng một prediction cho mỗi ảnh của toàn split.")
        cli = ["--pred", args.pred, "--splits-dir", str(Path(args.data) / "splits"),
               "--split", args.split, "--name", args.name, "--out", args.out]
        if args.checkpoint:
            run = json.loads((Path(args.checkpoint) / "run.json").read_text(encoding="utf-8"))
            if run["smoke"]:
                parser.error("Không chấm evidence từ checkpoint smoke.")
            version = json.loads((Path(args.data) / "splits/meta.json").read_text(encoding="utf-8"))["dataset_version"]
            if run["dataset_version"] != version:
                parser.error("dataset_version không khớp checkpoint.")
            for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
                cli.extend(["--meta", f"{key}={run[key]}"])
            cfg = run["config"]
            cli.extend(["--meta", f"base_setup={cfg['name']}", "--meta", f"train_fraction={cfg['train_fraction']}"])
        elif args.name.upper() != "E0":
            parser.error("Setup có train cần --checkpoint để lấy meta thật.")
        score_main(cli)


if __name__ == "__main__":
    main()
