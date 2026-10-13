from .coordinates import objects_to_answer
from .data import load_image
from .prompts import user_message


def answer_labels(full_ids, prompt_ids):
    import torch
    n = prompt_ids.shape[1]
    if full_ids.shape[1] <= n or not torch.equal(full_ids[:, :n], prompt_ids):
        raise AssertionError("Chat template: prompt không phải tiền tố của full input.")
    labels = full_ids.clone()
    labels[:, :n] = -100
    return labels


def encode_example(processor, row, data_dir):
    user = user_message(load_image(row, data_dir))
    answer = objects_to_answer(row["objects"], row["width"], row["height"])
    kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt = processor.apply_chat_template([user], add_generation_prompt=True, **kwargs)
    full = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
        add_generation_prompt=False, **kwargs)
    full["labels"] = answer_labels(full["input_ids"], prompt["input_ids"])
    # Không đưa metadata processor vào model.
    return {k: v for k, v in full.items() if k in {
        "input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw", "labels"}}


def collate_encoded(examples, pad_token_id):
    import torch
    keys = set(examples[0])
    if any(set(e) != keys for e in examples):
        raise AssertionError("Các sample không có cùng trường processor.")
    length = max(e["input_ids"].shape[1] for e in examples)
    result = {}
    for key in keys:
        if key in {"pixel_values", "image_grid_thw"}:
            result[key] = torch.cat([e[key] for e in examples], dim=0)
        else:
            fill = -100 if key == "labels" else pad_token_id if key == "input_ids" else 0
            result[key] = torch.cat([
                torch.nn.functional.pad(e[key], (0, length-e[key].shape[1]), value=fill)
                for e in examples], dim=0)
    result["labels"][result["attention_mask"] == 0] = -100
    return result


class SFTDataset:
    def __init__(self, rows, processor, data_dir):
        self.rows, self.processor, self.data_dir = rows, processor, data_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return encode_example(self.processor, self.rows[index], self.data_dir)


def inspect_encoding(processor, row, encoded):
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    count = int((encoded["input_ids"] == image_id).sum())
    if (row["width"], row["height"]) == (640, 360):
        assert count == 220, f"Token ảnh {count} != 220; kiểm tra resize processor."
    ids = encoded["labels"][encoded["labels"] != -100].tolist()
    decoded = processor.tokenizer.decode(ids, skip_special_tokens=False)
    expected = objects_to_answer(row["objects"], row["width"], row["height"])
    assert decoded.startswith(expected) and "<|im_end|>" in decoded[len(expected):], decoded
    assert decoded[len(expected):].replace("<|im_end|>", "").strip() == "", decoded
    print({"image_tokens": count, "supervised_tokens": len(ids), "decoded_labels": decoded})
