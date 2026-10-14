import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .config import write_json
from .setups import is_projector


def projector_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if is_projector(n)}


def save_checkpoint(model, processor, path, cfg, run):
    import torch
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)
    if cfg["train_projector"]:
        state = projector_state(model)
        assert state, "Không tìm thấy projector để lưu."
        torch.save(state, path / "projector.pt")
    write_json(path / "config.json", cfg)
    write_json(path / "run.json", run)
    write_json(path / "complete.json", {"complete": True})


def restore_projector(model, path, required):
    import torch
    path = Path(path) / "projector.pt"
    if not required:
        if path.exists():
            raise ValueError("Checkpoint chứa projector nhưng config không mở projector.")
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(model.named_parameters())
    expected = {n for n in params if is_projector(n)}
    if set(state) != expected:
        raise AssertionError("Tên projector checkpoint không khớp model.")
    with torch.no_grad():
        for n, value in state.items():
            if params[n].shape != value.shape:
                raise AssertionError(n)
            params[n].data = params[n].data.to(torch.float32)
            params[n].copy_(value)


def load_checkpoint(base, path):
    from peft import PeftModel
    path = Path(path)
    if not (path / "complete.json").exists():
        raise ValueError("Checkpoint chưa lưu xong.")
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = PeftModel.from_pretrained(base, path, is_trainable=False)
    restore_projector(model, path, cfg["train_projector"])
    return model


def verify_reload(original, restored, include_projector):
    import torch
    other = dict(restored.named_parameters())
    checked = 0
    for name, value in original.named_parameters():
        if "lora_" in name or (include_projector and is_projector(name)):
            assert name in other and torch.equal(value.detach().cpu(), other[name].detach().cpu()), name
            checked += 1
    assert checked > 0
    return checked


def backup_run(source, backup_root):
    if not backup_root:
        return None
    source = Path(source)
    destination = Path(backup_root) / source.name
    if destination.resolve() == source.resolve():
        raise ValueError("Thư mục backup phải khác thư mục chạy.")
    checkpoints = sorted(p for p in source.glob("step-*")
                         if p.is_dir() and (p / "complete.json").is_file())
    latest = checkpoints[-1] if checkpoints else None
    status = {"time_utc": datetime.now(timezone.utc).isoformat(),
              "destination": str(destination), "checkpoint": latest.name if latest else None}
    try:
        destination.mkdir(parents=True, exist_ok=True)
        # Không chép lại toàn bộ step-0100 khi vừa lưu step-0200.
        if latest:
            target = destination / latest.name
            target.mkdir(parents=True, exist_ok=True)
            for file in latest.rglob("*"):
                if file.is_file() and file.name != "complete.json":
                    copied = target / file.relative_to(latest)
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, copied)
                    if copied.stat().st_size != file.stat().st_size:
                        raise OSError(f"Backup size mismatch: {copied}")
            # Marker chỉ được chép sau khi tất cả file checkpoint chép thành công.
            shutil.copy2(latest / "complete.json", target / "complete.json")
        for file in source.iterdir():
            if file.is_file() and file.name != "backup_status.json":
                shutil.copy2(file, destination / file.name)
        status["ok"] = True
        print("BACKUP OK:", destination, status["checkpoint"], flush=True)
    except (OSError, shutil.Error) as error:
        status.update(ok=False, error=str(error))
        print("BACKUP FAILED - training continues with LOCAL checkpoint; back up before resetting:",
              error, flush=True)
    # Lỗi Drive không được làm mất cả lượt train đã lưu local.
    write_json(source / "backup_status.json", status)
    return status["ok"]
