"""CPU-only plot: python -m src.plot_training_logs. Never loads a model."""
import json
import math
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path("report/logs/sampled_training_events.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for run, label in (("work/runs/e3_scale1024", "100% train (E3, 200 steps)"),
                       ("work/runs/e4_data10_step200", "10% train (E4, 200 steps)")):
        events = sorted((e for e in data["events"] if e["run"] == run), key=lambda e: e["step"])
        assert len(events) == 21 and events[-1]["step"] == 200
        assert all(math.isfinite(e[key]) for e in events for key in ("loss", "grad_norm", "lr"))
        for axis, key, title in zip(axes, ("loss", "grad_norm"), ("Training loss", "Gradient norm before clipping")):
            axis.plot([e["step"] for e in events], [e[key] for e in events], marker=".", label=label)
            axis.set(xlabel="Optimizer step", ylabel=key, title=title)
            axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Printed log samples: step 1 and every 10 steps; no smoothing", fontsize=10)
    output = Path("report/figures/training_log_samples.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
