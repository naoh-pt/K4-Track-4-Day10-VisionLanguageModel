"""Điều phối Colab, log trực tiếp và khôi phục file; không nạp model."""
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

def run(*args):
    logs = Path("work/logs")
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.log"
    command = [sys.executable, "-u", *map(str, args)]
    print(" ".join(command), "\nLog:", path, flush=True)
    with path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f"Exit code {code}; xem {path}")
    return path


def restore_from_drive(root, run_names):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Chưa mount Drive hoặc sai tài khoản: {root}")
    # Khôi phục kết quả, không ghi đè source mới bằng source backup cũ.
    for name in ("report", "work/predictions", "work/validation", "work/logs"):
        source = root / "artifacts" / name
        if source.is_dir():
            for file in source.rglob("*"):
                if file.is_file():
                    target = Path(name) / file.relative_to(source)
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, target)
    for name in run_names:
        source, target = root / "runs" / name, Path("work/runs") / name
        if not source.is_dir():
            raise FileNotFoundError(f"Thiếu run backup: {source}")
        # Chép phần còn thiếu, giữ checkpoint/log đang có trong runtime.
        for file in source.rglob("*"):
            if file.is_file():
                destination = target / file.relative_to(source)
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, destination)
        print("Restored:", name)


def training_overrides(cfg):
    args = []
    if (cfg["batch_size"], cfg["grad_accum"]) == (1, 8):
        args += ["--batch-size", "1", "--grad-accum", "8"]
    elif (cfg["batch_size"], cfg["grad_accum"]) != (2, 4):
        raise ValueError("Batch/accum của đối chứng chưa được hỗ trợ.")
    if "amp_init_scale" in cfg:
        args += ["--amp-init-scale", str(cfg["amp_init_scale"])]
    return args


def e4_overrides(checkpoint, config_path):
    from .config import load_config
    cfg = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
    # Bảo đảm chỉ fraction thay đổi; không âm thầm bỏ tham số của đối chứng.
    expected = load_config(config_path)
    if (cfg["batch_size"], cfg["grad_accum"]) == (1, 8):
        expected.update(batch_size=1, grad_accum=8)
    if "amp_init_scale" in cfg:
        expected["amp_init_scale"] = cfg["amp_init_scale"]
    if expected != cfg:
        raise ValueError("Config nguồn khác checkpoint đối chứng; dừng để đối chiếu trước E4.")
    return training_overrides(cfg)
