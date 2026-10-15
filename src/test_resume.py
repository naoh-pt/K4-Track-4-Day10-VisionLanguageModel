"""CPU fixtures only: python -m unittest src.test_resume -v. No Qwen/processor downloads."""
import copy
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import torch

from .checkpoints import (backup_run, restore_run_archive, save_checkpoint,
                          verify_checkpoint)
from .modeling import parameter_report
from .resume import BatchStream, training_state, restore_training_state


class ResumeTests(unittest.TestCase):
    def test_batch_order_matches_original_loader_across_epochs(self):
        from torch.utils.data import DataLoader
        for size, batch in ((7, 3), (8, 2), (1, 2)):
            loader = DataLoader(list(range(size)), batch_size=batch, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(42))
            stream = BatchStream(size, batch, 42)
            for _ in range(3):
                expected = [b.tolist() for b in loader]
                self.assertEqual(expected, [stream.next_indices() for _ in expected])

    def test_resume_matches_uninterrupted_dropout_optimizer_scheduler_rng(self):
        def make():
            model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.3), torch.nn.Linear(4, 1))
            opt = torch.optim.AdamW(model.parameters(), lr=.01)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: .5*(1+math.cos(s/12)))
            scaler = torch.amp.GradScaler("cpu", init_scale=128., growth_interval=2)
            return model, opt, sched, scaler, BatchStream(7, 2, 42)

        def advance(parts, start, end):
            model, opt, sched, scaler, stream = parts
            batches, lrs = [], []
            for _ in range(start, end):
                opt.zero_grad(set_to_none=True)
                for _ in range(2):
                    ids = stream.next_indices()
                    batches.append(ids)
                    x = torch.tensor([[i/7, random.random(), 1.] for i in ids])
                    loss = model(x).square().mean()/2
                    scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                scaler.step(opt)
                scaler.update()
                sched.step()
                lrs.append(opt.param_groups[0]["lr"])
            return batches, lrs

        torch.manual_seed(123)
        random.seed(456)
        initial = make()
        before = copy.deepcopy(initial[0].state_dict())
        first_batches, first_lrs = advance(initial, 0, 4)
        model, opt, sched, scaler, stream = initial
        saved_weights = copy.deepcopy(model.state_dict())
        state = training_state(opt, sched, scaler, stream, step=4, fingerprint="fixture",
                               trainable_names=list(dict(model.named_parameters())), smoke=False, target_steps=9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save(state, path)
            expected_batches, expected_lrs = advance(initial, 4, 9)
            expected_weights = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(opt.state_dict())
            expected_scale = scaler.get_scale()
            # Simulate new process: initialization consumes RNG before restore.
            torch.manual_seed(999)
            random.seed(888)
            resumed = make()
            resumed[0].load_state_dict(saved_weights)
            restored = torch.load(path, map_location="cpu", weights_only=True)
            step = restore_training_state(restored, *resumed[1:], fingerprint="fixture",
                                          trainable_names=list(dict(resumed[0].named_parameters())),
                                          smoke=False, target_steps=9)
            self.assertEqual(step, 4)
            actual_batches, actual_lrs = advance(resumed, 4, 9)
            self.assertEqual(expected_batches, actual_batches)
            self.assertEqual(expected_lrs, actual_lrs)
            self.assertEqual(expected_scale, resumed[3].get_scale())
            for n, weight in resumed[0].state_dict().items():
                self.assertTrue(torch.equal(expected_weights[n], weight), n)
            for key, item in expected_optimizer["state"].items():
                for name, value in item.items():
                    self.assertTrue(torch.equal(value, resumed[1].state_dict()["state"][key][name]))
            self.assertFalse(torch.equal(before["0.weight"], expected_weights["0.weight"]))
            for invalid in ({"fingerprint": "other"}, {"target_steps": 10}, {"smoke": True}):
                options = dict(fingerprint="fixture", trainable_names=list(dict(resumed[0].named_parameters())),
                               smoke=False, target_steps=9)
                options.update(invalid)
                with self.assertRaises(ValueError):
                    restore_training_state(restored, *resumed[1:], **options)

    @staticmethod
    def fixture_checkpoint(root):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = torch.nn.Module()
                self.visual.merger = torch.nn.Linear(2, 3)

            def save_pretrained(self, path, **kwargs):
                (Path(path) / "adapter_model.safetensors").write_bytes(b"tiny adapter fixture")

        class TinyProcessor:
            def save_pretrained(self, path):
                (Path(path) / "processor_config.json").write_text("{}")

        root.mkdir(parents=True)
        (root / "train_ids.json").write_text('["fixture"]')
        cp = root / "step-0004"
        save_checkpoint(TinyModel(), TinyProcessor(), cp, {"train_projector": True}, {"steps": 4},
                        training_state={"format_version": 1, "step": 4})
        return cp

    def test_atomic_manifest_zip_roundtrip_and_nonfatal_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cp = self.fixture_checkpoint(root / "run")
            verify_checkpoint(cp, require_resume=True)
            self.assertTrue(backup_run(cp.parent, root / "drive"))
            archive = next((root / "drive").glob("*.zip"))
            self.assertTrue(backup_run(cp.parent, root / "drive"))
            self.assertEqual(len(list((root / "drive").glob("*__latest.zip"))), 1)
            previous_archive = archive.read_bytes()
            restored = restore_run_archive(archive, root / "restored")
            self.assertEqual((cp / "manifest.json").read_bytes(), (restored / "manifest.json").read_bytes())
            self.assertTrue((restored / "training_state.pt").exists())
            self.assertTrue((restored.parent / "train_ids.json").exists())
            with patch("src.checkpoints.shutil.copy2", side_effect=OSError("fixture Drive failure")):
                self.assertFalse(backup_run(cp.parent, root / "drive"))
            self.assertEqual(archive.read_bytes(), previous_archive)
            self.assertTrue((cp / "complete.json").exists())
            (restored / "training_state.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                verify_checkpoint(restored, require_resume=True)

    def test_reject_legacy_resume_and_unsafe_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "complete.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Checkpoint cũ"):
                verify_checkpoint(root, require_resume=True)
            archive = root / "bad.zip"
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", "fixture")
            with self.assertRaisesRegex(ValueError, "ngoài thư mục"):
                restore_run_archive(archive, root / "out")
            self.assertFalse((root / "escape.txt").exists())

    def test_architecture_counts_original_shape_for_packed_weights(self):
        class Params4bit:
            quant_state = SimpleNamespace(shape=(4, 5))
            def numel(self):
                return 10  # Packed storage is not the logical parameter count.
        class Fixture:
            config = SimpleNamespace(_commit_hash="fixture")
            def named_parameters(self):
                return iter([("model.language_model.layers.0.weight", Params4bit()),
                             ("model.visual.merger.weight", torch.zeros(2, 3)),
                             ("model.visual.deepstack_merger_list.0.weight", torch.zeros(3, 3))])
        result = parameter_report(Fixture())
        self.assertEqual(result["groups"]["language_model"], 20)
        self.assertEqual(result["projector_total"], 15)
        self.assertEqual(result["total_unique_parameters"], 35)


if __name__ == "__main__":
    unittest.main()
