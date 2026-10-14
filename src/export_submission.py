"""Colab CPU: phân tích prediction có thật, vẽ 6 ảnh và đóng gói file nhỏ để nộp.

python -m src.export_submission --pred work/predictions/e3_step0200_test.jsonl
Không nạp model, không đóng gói dataset hoặc checkpoint.
"""
import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .config import write_json
from .data import read_jsonl, read_split
from .visualize import draw_examples


def export(prediction, data_dir="data", output="work/submission_artifacts.zip"):
    from scripts.score import _gt_boxes, _match, evaluate, parse_output
    rows = read_split(data_dir, "test")
    raw_rows = read_jsonl(prediction)
    preds = {r["image_id"]: r["raw"] for r in raw_rows}
    assert len(preds) == len(raw_rows) == len(rows) and set(preds) == {r["image_id"] for r in rows}
    candidates = []
    for row in rows:
        parsed = parse_output(preds[row["image_id"]])
        per, _ = _match(_gt_boxes(row), parsed or [])
        tp, fp, fn = [sum(v[i] for v in per.values()) for i in range(3)]
        candidates.append({"image_id": row["image_id"], "tp": tp, "fp": fp, "fn": fn,
                           "valid_json": parsed is not None})
    # Ưu tiên ảnh đúng hoàn toàn; nếu ít hơn 3, dùng ảnh có box TP và mô tả đúng.
    positive = sorted((c for c in candidates if c["tp"] > 0),
                      key=lambda c: (c["fp"]+c["fn"], -c["tp"], c["image_id"]))[:3]
    selected = {c["image_id"] for c in positive}
    errors = sorted((c for c in candidates if c["image_id"] not in selected and c["fp"]+c["fn"] > 0),
                    key=lambda c: (-(c["fp"]+c["fn"]), c["image_id"]))[:3]
    chosen = positive + errors
    assert len(chosen) == 6, "Không đủ 6 mẫu; cần chọn thủ công."
    figure_dir = "report/figures/e3_step0200"
    paths = draw_examples(data_dir=data_dir, split="test", prediction=prediction,
                          out=figure_dir, count=6, image_ids=[c["image_id"] for c in chosen])
    for c, path in zip(chosen, paths):
        c["figure"] = path.as_posix()
        c["caption"] = (f"GT xanh, dự đoán đỏ; TP={c['tp']}, FP={c['fp']}, FN={c['fn']}. "
                        + ("Đúng toàn bộ box." if c["fp"]+c["fn"] == 0 else "Còn box thừa hoặc bỏ sót.")
                        + (" JSON không hợp lệ." if not c["valid_json"] else ""))
    write_json("report/analysis/figure_manifest.json", {"prediction": str(prediction), "examples": chosen})
    correct = evaluate(rows, preds, coord="norm1000")
    wrong = evaluate(rows, preds, coord="pixel")
    dataset_version = json.loads((Path(data_dir) / "splits/meta.json").read_text())["dataset_version"]
    write_json("report/analysis/coord_scale_check.json", {
        "dataset_version": dataset_version, "split": "test", "prediction": str(prediction),
        "description": "Cùng prediction E3, không suy luận/train lại; cố ý chấm sai thang để phân tích lỗi.",
        "correct_norm1000_f1": correct["micro"]["f1"], "wrong_pixel_f1": wrong["micro"]["f1"],
        "parse_valid_rate": correct["parse_valid_rate"]})
    for name in ("e1", "e2", "e3"):
        source = Path("work/validation") / f"{name}.json"
        if source.exists():
            write_json(Path("report/validation") / source.name, json.loads(source.read_text()))
    write_json("report/analysis/dataset_meta.json", json.loads((Path(data_dir) / "splits/meta.json").read_text()))
    evidence = Path("report/evidence/e3.json")
    if evidence.exists():
        expected = json.loads(evidence.read_text())
        assert expected["dataset_version"] == dataset_version
        assert abs(expected["micro"]["f1"] - correct["micro"]["f1"]) < 1e-12
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in Path("report").rglob("*") if p.is_file() and p.suffix in {".json", ".png"}]
    with ZipFile(output, "x", ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.as_posix())
    print("Exported:", output, "files:", len(files))
    print("Coordinate check:", correct["micro"]["f1"], "vs wrong pixel:", wrong["micro"]["f1"])
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="work/submission_artifacts.zip")
    args = parser.parse_args()
    export(args.pred, args.data, args.out)
