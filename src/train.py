import gc
import json
import math
import random
import time
from functools import partial
from pathlib import Path

from .checkpoints import backup_run, load_checkpoint, save_checkpoint
from .config import fingerprint, write_json
from .data import read_split, subset_train
from .modeling import load_base, require_colab_gpu
from .setups import apply_setup, is_projector
from .sft import SFTDataset, collate_encoded, inspect_encoding


def train(cfg, data_dir, out, smoke=False, smoke_steps=2, backup_root=None, smoke_dir=None):
    require_colab_gpu()
    import torch
    from torch.utils.data import DataLoader
    from peft.utils.save_and_load import get_peft_model_state_dict

    if cfg["name"] == "E0":
        raise ValueError("E0 không train.")
    data_dir, out = Path(data_dir), Path(out)
    dataset_version = json.loads((data_dir / "splits/meta.json").read_text(encoding="utf-8"))["dataset_version"]
    signature = fingerprint(cfg, dataset_version)
    if not smoke:
        marker = json.loads((Path(smoke_dir or "") / "smoke_pass.json").read_text(encoding="utf-8"))
        if marker.get("fingerprint") != signature:
            raise ValueError("Smoke test không khớp cấu hình/dataset. Chạy smoke lại cho cấu hình này.")
    if smoke_steps < 1:
        raise ValueError("smoke_steps phải dương.")
    out.mkdir(parents=True, exist_ok=False)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    rows = subset_train(read_split(data_dir, "train"), cfg["train_fraction"], cfg["seed"])
    if smoke:
        rows = rows[:8]
    write_json(out / "train_ids.json", [r["image_id"] for r in rows])
    model, processor = load_base(cfg)
    model, setup = apply_setup(model, cfg)
    write_json(out / "setup.json", setup)
    dataset = SFTDataset(rows, processor, data_dir)
    inspect_encoding(processor, rows[0], dataset[0])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
                        collate_fn=partial(collate_encoded, pad_token_id=processor.tokenizer.pad_token_id),
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=0)
    # Smoke dùng cùng lịch LR với run đầy đủ nhưng dừng sớm.
    warmup = max(1, math.ceil(cfg["max_steps"] * 0.05))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1, (step-warmup) / max(1, cfg["max_steps"]-warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", init_scale=cfg.get("amp_init_scale", 65536.0))
    print({"amp_init_scale": scaler.get_scale()}, flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    iterator = iter(loader)
    steps = smoke_steps if smoke else cfg["max_steps"]
    projector_sample = None
    adapter_sample = None
    for name, p in model.named_parameters():
        if p.requires_grad and is_projector(name) and projector_sample is None:
            projector_sample = (name, p.detach().flatten()[:4096].cpu().clone())
        if "lora_B" in name and adapter_sample is None:
            adapter_sample = (name, p.detach().cpu().clone())
    log_path = out / "train_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    with log_path.open("x", encoding="utf-8") as log:
        for step in range(1, steps+1):
            loss_sum = 0.0
            for _ in range(cfg["grad_accum"]):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {k: v.to("cuda:0") for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Loss không hữu hạn ở step {step}.")
                loss_sum += float(loss.detach()) / cfg["grad_accum"]
                scaler.scale(loss / cfg["grad_accum"]).backward()
            scaler.unscale_(optimizer)
            try:
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            except RuntimeError:
                bad = [n for n, p in model.named_parameters()
                       if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
                diagnostic = {"step": step, "loss": loss_sum, "amp_scale": scaler.get_scale(),
                              "nonfinite_gradient_params": bad[:20], "nonfinite_param_count": len(bad)}
                write_json(out / "failure.json", diagnostic)
                print(diagnostic, flush=True)
                raise FloatingPointError(f"Gradient không hữu hạn ở step {step}; xem failure.json.")
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Gradient không hữu hạn ở step {step}; dừng và gửi log.")
            if smoke and step == 1:
                absent = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
                assert not absent, f"Tham số trainable không có gradient: {absent[:10]}"
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            event = {"step": step, "loss": loss_sum, "grad_norm": float(norm),
                     "lr": optimizer.param_groups[0]["lr"], "amp_scale": scaler.get_scale()}
            log.write(json.dumps(event) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0 or step == steps:
                print(event, flush=True)
            if step % 100 == 0 or step == steps:
                torch.cuda.synchronize()
                run = {"trainable_params": setup["trainable_params"],
                       "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
                       "train_minutes": (time.perf_counter()-started) / 60,
                       "dataset_version": dataset_version, "steps": step, "smoke": smoke,
                       "fingerprint": signature, "config": cfg}
                checkpoint = out / f"step-{step:04d}"
                save_checkpoint(model, processor, checkpoint, cfg, run)
                write_json(out / "run.json", run)
                backup_run(out, backup_root)
    if smoke:
        current = dict(model.named_parameters())
        assert adapter_sample is not None
        n, before = adapter_sample
        assert not torch.equal(before, current[n].detach().cpu()), "Adapter chưa thay đổi."
        if cfg["train_projector"]:
            n, before = projector_sample
            assert not torch.equal(before, current[n].detach().flatten()[:4096].cpu()), "Projector chưa thay đổi."
        # Lưu snapshot CPU nhỏ cho adapter; projector đã có file checkpoint riêng.
        expected_adapter = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
        del current, model, optimizer, scheduler, params, batch, loss
        gc.collect()
        torch.cuda.empty_cache()
        # Nạp lại model gốc để kiểm tra round-trip trong giới hạn 16 GB.
        base, processor = load_base(cfg)
        restored = load_checkpoint(base, checkpoint)
        actual = get_peft_model_state_dict(restored)
        assert set(expected_adapter) == set(actual)
        assert all(torch.equal(v, actual[k].detach().cpu()) for k, v in expected_adapter.items())
        if cfg["train_projector"]:
            saved = torch.load(checkpoint / "projector.pt", map_location="cpu", weights_only=True)
            named = dict(restored.named_parameters())
            assert all(torch.equal(v, named[k].detach().cpu()) for k, v in saved.items())
            del saved, named
        from .inference import predict
        from scripts.score import evaluate
        from .data import read_jsonl
        val = read_split(data_dir, "val")[:2]
        prediction = predict(restored, processor, val, data_dir, out / "smoke_val.jsonl", batch_size=2)
        metrics = evaluate(val, {r["image_id"]: r["raw"] for r in read_jsonl(prediction)}, n_boot=50)
        write_json(out / "smoke_metrics.json", metrics)
        print("Smoke output:", read_jsonl(prediction), flush=True)
        # Vài bước chưa đủ để đòi chất lượng/JSON hoàn hảo; smoke xác nhận đường chạy.
        write_json(out / "smoke_pass.json", {"fingerprint": signature, "steps": steps,
                   "adapter_roundtrip": True, "projector_roundtrip": bool(cfg["train_projector"])})
        backup_run(out, backup_root)
        print("SMOKE PASS — có thể chạy full với --smoke-dir", out, flush=True)
    print("Checkpoint:", checkpoint, flush=True)
    return checkpoint
