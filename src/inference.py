import json
from pathlib import Path

from .data import load_image
from .prompts import user_message


def predict(model, processor, rows, data_dir, output, batch_size=4, max_new_tokens=256):
    import torch
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Đã có prediction: {output}; dùng tên run mới để giữ kết quả.")
    output.parent.mkdir(parents=True, exist_ok=True)
    processor.tokenizer.padding_side = "left"
    model.eval()
    model.config.use_cache = True
    with output.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start+batch_size]
            messages = [[user_message(load_image(r, data_dir))] for r in batch]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda:0") for k, v in inputs.items() if k in {
                "input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}}
            with torch.autocast("cuda", dtype=torch.float16):
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            texts = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, raw in zip(batch, texts):
                stream.write(json.dumps({"image_id": row["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"predict {min(start+batch_size,len(rows))}/{len(rows)}", flush=True)
    return output
