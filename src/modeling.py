import importlib.util

SKIP_MODULES = ["model.visual", "lm_head"]


def require_colab_gpu():
    # Guard chạy trước mọi from_pretrained, kể cả processor.
    try:
        in_colab = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        in_colab = False
    if not in_colab:
        raise RuntimeError("Chỉ nạp model/processor trong Google Colab GPU; local không được phép.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Chọn Runtime > Change runtime type > T4 GPU trước khi chạy.")


def load_base(cfg):
    require_colab_gpu()
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        # Transformers 5.19 so prefix/regex/suffix trên tên đầy đủ: "visual"
        # không khớp "model.visual.blocks...". Phải ghi đúng prefix cây model.
        bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=SKIP_MODULES)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], quantization_config=quant, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    for name, module in model.named_modules():
        if ("visual" in name or name == "lm_head") and "4bit" in type(module).__name__.lower():
            raise AssertionError(f"Module không được lượng tử hóa: {name}")
    assert any("language_model" in n and "4bit" in type(m).__name__.lower()
               for n, m in model.named_modules()), "LLM phải được lượng tử hóa 4-bit."
    print("Quantization OK: LLM 4-bit; visual và lm_head không bị lượng tử hóa.", flush=True)
    return model, processor
