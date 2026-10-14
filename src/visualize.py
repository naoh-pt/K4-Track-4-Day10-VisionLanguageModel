from pathlib import Path

from .coordinates import objects_to_answer, to_pixels
from .data import load_image, read_jsonl, read_split


def draw_examples(data_dir="data", split="val", prediction=None, coord="norm1000", out="report/figures", count=6,
                  image_ids=None):
    from PIL import ImageDraw
    from scripts.score import parse_output
    rows = read_split(data_dir, split)
    if image_ids is not None:
        by_id = {r["image_id"]: r for r in rows}
        rows = [by_id[image_id] for image_id in image_ids]
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(prediction)} if prediction else {}
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in rows[:count]:
        image = load_image(row, data_dir)
        draw = ImageDraw.Draw(image)
        gt = parse_output(objects_to_answer(row["objects"], row["width"], row["height"]))
        pred = parse_output(preds.get(row["image_id"])) if prediction else []
        for boxes, color, prefix, scale in ((gt, "lime", "GT", "norm1000"), (pred or [], "red", "PRED", coord)):
            for box in boxes:
                xy = to_pixels(box["bbox"], row["width"], row["height"], scale)
                draw.rectangle(xy, outline=color, width=2)
                draw.text((xy[0], max(14, xy[1])-12), f"{prefix}: {box['label']}", fill=color)
        draw.rectangle((0, 0, image.width, 14), fill="black")
        caption = "GT green / PRED red" + (" / INVALID JSON" if prediction and pred is None else "")
        draw.text((2, 1), caption, fill="white")
        path = output / f"{split}_{row['image_id']}.png"
        image.save(path)
        paths.append(path)
    return paths
