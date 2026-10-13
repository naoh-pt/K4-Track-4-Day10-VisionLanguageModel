import json
import shutil
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
    if backup_root:
        source = Path(source)
        destination = Path(backup_root) / source.name
        if destination.resolve() == source.resolve():
            raise ValueError("Thư mục backup phải khác thư mục chạy.")
        shutil.copytree(source, destination, dirs_exist_ok=True)
