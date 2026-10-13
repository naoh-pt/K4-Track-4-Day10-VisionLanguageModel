import json

CLASSES = ("pothole", "crack", "manhole")


def yolo_to_box(label, yolo, width=640, height=360, coord="norm1000"):
    if label not in CLASSES or coord not in {"norm1000", "pixel"}:
        raise ValueError("Nhãn hoặc thang tọa độ không hợp lệ.")
    if len(yolo) != 4 or not all(0 <= v <= 1 for v in yolo):
        raise ValueError("YOLO cần bốn số trong [0,1].")
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    bounds = (sx, sy, sx, sy)
    values = ((xc-w/2)*sx, (yc-h/2)*sy, (xc+w/2)*sx, (yc+h/2)*sy)
    return {"label": label, "bbox_2d": [max(0, min(b, round(v))) for v, b in zip(values, bounds)]}


def objects_to_answer(objects, width=640, height=360, coord="norm1000"):
    boxes = [yolo_to_box(o["label"], o["yolo"], width, height, coord) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def to_pixels(box, width, height, coord="norm1000"):
    if coord == "pixel":
        return tuple(box)
    return tuple(v * (width if i % 2 == 0 else height) / 1000 for i, v in enumerate(box))
