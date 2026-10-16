"""Kiểm tra CPU nhỏ: python -m unittest src.test_lightweight -v. Không có model thật."""
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .checkpoints import projector_state, restore_projector
from .config import load_config, fingerprint
from .coordinates import objects_to_answer, to_pixels, yolo_to_box
from .data import subset_train
from .modeling import SKIP_MODULES, require_colab_gpu
from .setups import select_targets
from .sft import answer_labels, collate_encoded


class LightweightTests(unittest.TestCase):
    def test_export_uses_real_predictions_and_excludes_weights(self):
        from PIL import Image
        from zipfile import ZipFile
        from .export_submission import export
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                Path("data/images").mkdir(parents=True)
                Path("data/splits").mkdir(parents=True)
                rows, predictions = [], []
                for i in range(6):
                    image_path = f"images/{i}.png"
                    Image.new("RGB", (640, 360)).save(Path("data") / image_path)
                    objects = [{"label": "pothole", "yolo": [.5, .5, .2, .2]}]
                    rows.append({"image_id": str(i), "image_path": image_path,
                                 "width": 640, "height": 360, "objects": objects})
                    predictions.append({"image_id": str(i), "raw": objects_to_answer(objects) if i < 3 else "[]"})
                Path("data/splits/test.jsonl").write_text("\n".join(map(json.dumps, rows)))
                Path("data/splits/meta.json").write_text(json.dumps({"dataset_version": "synthetic"}))
                Path("pred.jsonl").write_text("\n".join(map(json.dumps, predictions)))
                output = export("pred.jsonl")
                manifest = json.loads(Path("report/analysis/figure_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(len(manifest["examples"]), 6)
                self.assertEqual(sum(e["tp"] > 0 for e in manifest["examples"]), 3)
                check = json.loads(Path("report/analysis/coord_scale_check.json").read_text(encoding="utf-8"))
                self.assertGreater(check["correct_norm1000_f1"], check["wrong_pixel_f1"])
                with ZipFile(output) as z:
                    self.assertEqual(sum(n.endswith(".png") for n in z.namelist()), 6)
                    self.assertTrue(all(n.startswith("report/") for n in z.namelist()))
                Path("report/extended/validation").mkdir(parents=True)
                Path("work/validation").mkdir(parents=True)
                Path("report/extended/validation/e3.json").write_text('{"steps":300}')
                Path("work/validation/e3.json").write_text('{"steps":200}')
                extended = export("pred.jsonl", output="work/extended.zip",
                                  report_root="report/extended", figure_tag="e3_step0300")
                self.assertEqual(json.loads(Path("report/extended/validation/e3.json").read_text()), {"steps": 300})
                with ZipFile(extended) as z:
                    self.assertTrue(all(n.startswith("report/extended/") for n in z.namelist()))
                    self.assertEqual(sum("figures/e3_step0300/" in n for n in z.namelist()), 6)
            finally:
                os.chdir(previous)

    def test_recorded_early_stop_preserves_original_lr_horizon(self):
        from .run import main
        with patch("sys.argv", ["src.run", "train", "--config", "configs/e3.yaml",
                                "--out", "unused", "--fraction", "0.1", "--amp-init-scale", "1024",
                                "--stop-after", "200", "--smoke"]), patch("src.train.train") as train:
            main()
        cfg = train.call_args.args[0]
        self.assertEqual(cfg["max_steps"], 300)
        self.assertEqual(cfg["stop_after"], 200)
        self.assertEqual(cfg["train_fraction"], .1)
        self.assertEqual(cfg["amp_init_scale"], 1024.)
        self.assertEqual((cfg["batch_size"], cfg["grad_accum"]), (2, 4))

    def test_backup_latest_only_and_drive_failure_is_nonfatal(self):
        from .checkpoints import backup_run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, backup = root / "run", root / "drive"
            for step in ("step-0100", "step-0200"):
                folder = source / step
                folder.mkdir(parents=True)
                (folder / "complete.json").write_text("{}")
                (folder / "projector.pt").write_bytes(b"tiny synthetic checkpoint")
            self.assertTrue(backup_run(source, backup))
            self.assertFalse((backup / "run/step-0100").exists())
            self.assertTrue((backup / "run/step-0200/complete.json").is_file())
            with patch("src.checkpoints.shutil.copy2", side_effect=OSError("synthetic Drive failure")):
                self.assertFalse(backup_run(source, root / "failed_drive"))
            self.assertFalse((root / "failed_drive/run/step-0200/complete.json").exists())
            self.assertTrue((source / "step-0200/projector.pt").is_file())
            status = json.loads((source / "backup_status.json").read_text(encoding="utf-8"))
            self.assertFalse(status["ok"])
            self.assertEqual(status["checkpoint"], "step-0200")

    def test_live_log_and_restore_do_not_overwrite_existing_files(self):
        from .colab_workflow import run, restore_from_drive
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                log = run("-c", "print('synthetic progress')")
                self.assertIn("synthetic progress", log.read_text(encoding="utf-8"))
                with self.assertRaisesRegex(RuntimeError, "Exit code 3"):
                    run("-c", "import sys; print('synthetic failure'); sys.exit(3)")
                root = Path("drive")
                source = root / "artifacts/report/evidence"
                source.mkdir(parents=True)
                (source / "existing.json").write_text("backup", encoding="utf-8")
                (source / "new.json").write_text("new", encoding="utf-8")
                destination = Path("report/evidence")
                destination.mkdir(parents=True)
                (destination / "existing.json").write_text("keep", encoding="utf-8")
                run_source = root / "runs/example"
                run_source.mkdir(parents=True)
                (run_source / "smoke_pass.json").write_text("{}", encoding="utf-8")
                restore_from_drive(root, ["example"])
                self.assertEqual((destination / "existing.json").read_text(), "keep")
                self.assertEqual((destination / "new.json").read_text(), "new")
                self.assertTrue(Path("work/runs/example/smoke_pass.json").exists())
            finally:
                os.chdir(previous)

    def test_e4_inherits_amp_scale_and_rejects_config_drift(self):
        from .colab_workflow import e4_overrides
        cfg = load_config("configs/e3.yaml")
        cfg["amp_init_scale"] = 1024.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            self.assertEqual(e4_overrides(directory, "configs/e3.yaml"), ["--amp-init-scale", "1024.0"])
            cfg["lr"] = 1e-4
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "khác checkpoint"):
                e4_overrides(directory, "configs/e3.yaml")

    def test_quantization_skip_full_module_paths(self):
        # Quy tắc should_convert_module của Transformers v5.19.0;
        # không import Transformers hoặc tạo model trên local.
        def skipped(name):
            return any(re.match(f"{key}\\.", name) or re.match(key, name) or name.endswith(key)
                       for key in SKIP_MODULES)
        for name in ("model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1",
                     "model.visual.deepstack_merger_list.0.linear_fc1", "lm_head"):
            self.assertTrue(skipped(name), name)
        self.assertFalse(skipped("model.language_model.layers.0.self_attn.q_proj"))

    def test_coordinate_roundtrip_and_empty(self):
        box = yolo_to_box("pothole", [.5, .5, .2, .2])
        self.assertEqual(box["bbox_2d"], [400, 400, 600, 600])
        self.assertEqual(to_pixels(box["bbox_2d"], 640, 360), (256, 144, 384, 216))
        self.assertEqual(yolo_to_box("crack", [0, 1, 1, 1])["bbox_2d"], [0, 500, 500, 1000])
        self.assertEqual(objects_to_answer([]), "[]")
        self.assertEqual(json.loads(objects_to_answer([
            {"label": "crack", "yolo": [.5, .8, .2, .2]},
            {"label": "pothole", "yolo": [.5, .2, .2, .2]}]))[0]["label"], "pothole")

    def test_prefix_loss_and_padding(self):
        prompt = torch.tensor([[10, 11, 12]])
        full = torch.tensor([[10, 11, 12, 20, 21]])
        labels = answer_labels(full, prompt)
        self.assertEqual(labels.tolist(), [[-100, -100, -100, 20, 21]])
        with self.assertRaises(AssertionError):
            answer_labels(full, torch.tensor([[99]]))
        a = dict(input_ids=full, labels=labels, attention_mask=torch.ones_like(full),
                 mm_token_type_ids=torch.tensor([[0, 1, 1, 0, 0]]),
                 pixel_values=torch.ones((3, 4)), image_grid_thw=torch.tensor([[1, 2, 2]]))
        b = dict(input_ids=full[:, :4], labels=labels[:, :4], attention_mask=torch.ones((1, 4), dtype=torch.long),
                 mm_token_type_ids=torch.tensor([[0, 1, 1, 0]]),
                 pixel_values=torch.ones((2, 4)), image_grid_thw=torch.tensor([[1, 2, 1]]))
        batch = collate_encoded([a, b], 0)
        self.assertEqual(batch["labels"][1, -1].item(), -100)
        self.assertEqual(batch["attention_mask"][1, -1].item(), 0)
        self.assertEqual(tuple(batch["pixel_values"].shape), (5, 4))
        self.assertEqual(tuple(batch["image_grid_thw"].shape), (2, 3))
        self.assertEqual(batch["mm_token_type_ids"][1, -1].item(), 0)

    def test_lora_targets_exclude_projector_and_unrelated(self):
        names = []
        for i in range(36):
            names.extend(f"model.language_model.layers.{i}.self_attn.{p}_proj" for p in "qkvo")
            names.extend(f"model.language_model.layers.{i}.mlp.{p}_proj" for p in ("gate", "up", "down"))
        for i in range(24):
            names.extend(f"model.visual.blocks.{i}.{p}" for p in (
                "attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"))
        names.extend(["model.visual.merger.linear_fc1", "lm_head", "fake.q_proj"])
        self.assertEqual(len(select_targets(names, "llm")), 252)
        self.assertEqual(len(select_targets(names, "llm+vit")), 348)

    def test_projector_roundtrip_tiny_cpu(self):
        # Cây giả vài chục tham số, không khởi tạo Qwen/processor.
        model = torch.nn.Module()
        model.visual = torch.nn.Module()
        model.visual.merger = torch.nn.Linear(2, 3)
        model.visual.deepstack_merger_list = torch.nn.ModuleList([torch.nn.Linear(2, 3)])
        original = projector_state(model)
        with tempfile.TemporaryDirectory() as directory:
            torch.save(original, Path(directory) / "projector.pt")
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()
            restore_projector(model, directory, required=True)
            self.assertTrue(all(torch.equal(v, projector_state(model)[n]) for n, v in original.items()))

    def test_config_subset_and_local_guard(self):
        cfg = load_config("configs/e2.yaml")
        self.assertTrue(cfg["train_projector"])
        self.assertEqual(cfg["lr"], .0002)
        changed = load_config("configs/e2.yaml", fraction=.1)
        self.assertNotEqual(fingerprint(cfg, "v1"), fingerprint(changed, "v1"))
        rows = [{"image_id": str(i)} for i in range(100)]
        small = subset_train(rows, .1, 42)
        self.assertEqual(len(small), 10)
        self.assertEqual(small, subset_train(list(reversed(rows)), .1, 42))
        with self.assertRaisesRegex(RuntimeError, "Colab"):
            require_colab_gpu()

    def test_score_cli_uses_original_scorer_and_actual_run_meta(self):
        from .run import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "data/splits"
            checkpoint = root / "checkpoint"
            splits.mkdir(parents=True)
            checkpoint.mkdir()
            entry = {"image_id": "synthetic", "width": 640, "height": 360,
                     "objects": [{"label": "pothole", "yolo": [.5, .5, .2, .2]}]}
            (splits / "val.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
            (splits / "meta.json").write_text(json.dumps({"dataset_version": "synthetic-test"}), encoding="utf-8")
            run = {"smoke": False, "dataset_version": "synthetic-test", "trainable_params": 16,
                   "peak_vram_gb": 1.25, "train_minutes": .5,
                   "config": {"name": "E1", "train_fraction": 1.0}}
            (checkpoint / "run.json").write_text(json.dumps(run), encoding="utf-8")
            pred = root / "pred.jsonl"
            pred.write_text(json.dumps({"image_id": "synthetic", "raw": objects_to_answer(entry["objects"])}) + "\n",
                            encoding="utf-8")
            out = root / "result.json"
            args = ["src.run", "score", "--pred", str(pred), "--data", str(root / "data"),
                    "--split", "val", "--name", "E1", "--checkpoint", str(checkpoint), "--out", str(out)]
            with patch("sys.argv", args):
                main()
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["micro"]["f1"], 1)
            self.assertEqual(result["meta"]["train_minutes"], .5)
            self.assertEqual(result["dataset_version"], "synthetic-test")
            pred.write_text("", encoding="utf-8")
            args[-1] = str(root / "missing.json")
            with patch("sys.argv", args), self.assertRaisesRegex(ValueError, "toàn split"):
                main()


if __name__ == "__main__":
    unittest.main()
