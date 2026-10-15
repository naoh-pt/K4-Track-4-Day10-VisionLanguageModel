import json
import shutil
import hashlib
import tempfile
import os
import uuid
from zipfile import ZipFile, ZIP_STORED
from datetime import datetime, timezone
from pathlib import Path

from .config import write_json
from .setups import is_projector


def projector_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if is_projector(n)}


def save_checkpoint(model, processor, path, cfg, run, training_state=None):
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}_", dir=path.parent))
    model.save_pretrained(staging, safe_serialization=True)
    processor.save_pretrained(staging)
    if cfg["train_projector"]:
        state = projector_state(model)
        assert state, "Không tìm thấy projector để lưu."
        torch.save(state, staging / "projector.pt")
    if training_state is not None:
        torch.save(training_state, staging / "training_state.pt")
    write_json(staging / "config.json", cfg)
    write_json(staging / "run.json", run)
    manifest = {p.name: sha256_file(p) for p in staging.iterdir() if p.is_file()}
    write_json(staging / "manifest.json", manifest)
    write_json(staging / "complete.json", {"complete": True, "resumable": training_state is not None})
    staging.rename(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path, require_resume=False):
    path = Path(path)
    if not (path / "complete.json").is_file():
        raise ValueError(f"Checkpoint chưa hoàn tất: {path}")
    manifest_path = path / "manifest.json"
    if require_resume and not manifest_path.exists():
        raise ValueError("Checkpoint cũ không có state/manifest để resume; cần run mới từ model gốc.")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(Path(name).name != name for name in manifest):
            raise ValueError("Tên file manifest không hợp lệ.")
        for name, expected in manifest.items():
            if not (path / name).is_file() or sha256_file(path / name) != expected:
                raise ValueError(f"Checksum checkpoint sai/thiếu: {name}")
        if require_resume and "training_state.pt" not in manifest:
            raise ValueError("Checkpoint thiếu optimizer/RNG state.")
    return path


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
    path = verify_checkpoint(path)
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
        if latest and (latest / "manifest.json").is_file():
            # Checkpoint mới lớn vì có optimizer; một ZIP tránh lỗi Drive với thư mục nhiều file.
            archive = make_run_archive(source, latest)
            destination = Path(backup_root) / f"{source.name}__latest.zip"
            destination.parent.mkdir(parents=True, exist_ok=True)
            uploading = destination.parent / f".{source.name}__upload_{uuid.uuid4().hex[:8]}.zip"
            shutil.copy2(archive, uploading)
            if sha256_file(uploading) != sha256_file(archive):
                raise OSError(f"Backup ZIP checksum mismatch: {uploading}")
            # Rolling backup của đúng run mới: chỉ thay bản trước sau khi upload được kiểm tra.
            # Không đụng checkpoint/archive của các run đã nộp.
            os.replace(uploading, destination)
            status.update(ok=True, archive=str(destination), format="zip")
            write_json(source / "backup_status.json", status)
            print("BACKUP ZIP OK:", destination, flush=True)
            return True
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


def make_run_archive(source, checkpoint=None):
    source = Path(source)
    if checkpoint is None:
        choices = sorted(p for p in source.glob("step-*") if (p / "complete.json").is_file())
        if not choices:
            raise ValueError("Không có checkpoint hoàn tất.")
        checkpoint = choices[-1]
    checkpoint = verify_checkpoint(checkpoint)
    archives = source.parent / ".backup_archives"
    archives.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive = archives / f"{source.name}__{checkpoint.name}__{stamp}_{uuid.uuid4().hex[:8]}.zip"
    with ZipFile(archive, "x", compression=ZIP_STORED, allowZip64=True) as bundle:
        for file in checkpoint.rglob("*"):
            if file.is_file():
                bundle.write(file, file.relative_to(source).as_posix())
        for file in source.iterdir():
            if file.is_file() and file.suffix in {".json", ".jsonl", ".txt"}:
                bundle.write(file, file.name)
    return archive


def restore_run_archive(archive, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".restore_", dir=out.parent))
    with ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not (staging / member.filename).resolve().is_relative_to(staging.resolve()):
                raise ValueError("ZIP có đường dẫn ngoài thư mục đích.")
        bundle.extractall(staging)
    checkpoints = sorted(p for p in staging.glob("step-*") if p.is_dir())
    if len(checkpoints) != 1:
        raise ValueError("ZIP phải có đúng một checkpoint.")
    checkpoint = verify_checkpoint(checkpoints[0], require_resume=True)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / checkpoint.name
    if destination.exists():
        verify_checkpoint(destination, require_resume=True)
        if (destination / "manifest.json").read_bytes() != (checkpoint / "manifest.json").read_bytes():
            raise ValueError("Checkpoint local khác ZIP; giữ cả hai, dùng thư mục out mới.")
    else:
        checkpoint.rename(destination)
    for file in staging.iterdir():
        if file.is_file() and not (out / file.name).exists():
            shutil.copy2(file, out / file.name)
    # Staging được giữ nếu có file trùng, không xóa dữ liệu người dùng.
    print("RESTORED resumable checkpoint:", destination, flush=True)
    return destination
