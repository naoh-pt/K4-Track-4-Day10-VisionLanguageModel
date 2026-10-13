import re

LLM = re.compile(r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)")
VIT = re.compile(r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))")


def is_projector(name):
    return "visual.merger." in name or "visual.deepstack_merger_list." in name


def select_targets(names, targets):
    if targets not in {"llm", "llm+vit"}:
        raise ValueError(targets)
    return [n for n in names if LLM.fullmatch(n) or (targets == "llm+vit" and VIT.fullmatch(n))]


def apply_setup(model, cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    names = select_targets([n for n, _ in model.named_modules()], cfg["lora"]["targets"])
    expected = 348 if cfg["lora"]["targets"] == "llm+vit" else 252
    if len(names) != expected:
        raise AssertionError(f"LoRA: {len(names)} != {expected}; gửi cây module để kiểm tra phiên bản.")
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], target_modules="(?:" + "|".join(re.escape(n) for n in names) + ")"))
    projector_params = 0
    for name, p in model.named_parameters():
        if is_projector(name) and cfg["train_projector"]:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
            projector_params += p.numel()
        if p.requires_grad:
            assert "lora_" in name or (cfg["train_projector"] and is_projector(name)), name
            # GradScaler không unscale gradient fp16; adapter và projector train ở fp32.
            p.data = p.data.to(torch.float32)
    if cfg["train_projector"]:
        assert 100_000_000 < projector_params < 120_000_000, projector_params
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {"matched_modules": len(names), "module_names": names,
               "trainable_params": count, "projector_params": projector_params,
               "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad]}
    print({k: v for k, v in summary.items() if not isinstance(v, list)})
    return model, summary
