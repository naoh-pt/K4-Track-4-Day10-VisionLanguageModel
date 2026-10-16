import json
from pathlib import Path


def results_table(directory="report/evidence"):
    rows = []
    for path in sorted(Path(directory).glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        m = d.get("meta", {})
        rows.append({"file": path.name, "split": d["split"], "F1": d["micro"]["f1"],
                     "CI95": d["f1_ci95"], "FP/image": d["fp_per_image"],
                     "parse_valid_rate": d["parse_valid_rate"],
                     **{c: d["per_class"][c]["f1"] for c in ("pothole", "crack", "manhole")},
                     **{k: m.get(k) for k in ("trainable_params", "peak_vram_gb", "train_minutes")}})
    return rows


def choose_by_val(directory="work/validation"):
    rows = []
    for name in ("e1", "e2", "e3"):
        d = json.loads((Path(directory) / f"{name}.json").read_text(encoding="utf-8"))
        assert d["split"] == "val" and d["missing_predictions"] == 0
        rows.append((d["micro"]["f1"], name))
    best = max(rows, key=lambda r: (r[0], -int(r[1][1])))
    print("Chọn đối chứng bằng F1 val:", rows, "=>", best[1])
    return best[1]


def plot_f1_ci(directory="report/evidence"):
    """Display existing scorer evidence; no models, no output files."""
    import matplotlib.pyplot as plt
    names = ["e0", "e1", "e2", "e3", "e4_data_size_0.1", "e4_data_size_1.0"]
    rows = {row["file"][:-5]: row for row in results_table(directory)}
    selected = [rows[name] for name in names]
    values = [row["F1"] for row in selected]
    errors = [[row["F1"] - row["CI95"][0] for row in selected],
              [row["CI95"][1] - row["F1"] for row in selected]]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(range(len(names)), values, yerr=errors, fmt="o", capsize=5,
                color="tab:blue", linestyle="none")
    ax.set_xticks(range(len(names)), ["E0", "E1", "E2", "E3", "E4: 10%", "E4: 100%\n(reuses E3)"])
    ax.set(ylabel="Micro F1", ylim=(0, max(row["CI95"][1] for row in selected) + .07),
           title="Test: 311 images — F1 and scorer bootstrap CI95")
    ax.grid(axis="y", alpha=.25)
    for x, value in enumerate(values):
        ax.annotate(f"{value:.3f}", (x, value), xytext=(9, 0), textcoords="offset points")
    fig.tight_layout()
    return fig
