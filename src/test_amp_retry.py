"""Tiny CPU tests; no pretrained model or downloads."""
import copy
import random
import unittest

import torch

from .amp_retry import backward_with_retry
from .resume import BatchStream


class AmpRetryTests(unittest.TestCase):
    def run_window(self, scale, failures, max_retries=6):
        torch.manual_seed(42)
        random.seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(.25), torch.nn.Linear(3, 1))
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, 1, .9)
        scaler = torch.amp.GradScaler("cpu", init_scale=scale)
        stream = BatchStream(9, 2, 42)
        initial = copy.deepcopy(model.state_dict())
        attempts, events = [], []

        def compute():
            attempt = []
            for _ in range(2):
                ids = stream.next_indices()
                x = torch.tensor([[i / 9, random.random()] for i in ids])
                loss = model(x).square().mean() / 2
                scaler.scale(loss).backward()
                attempt.append((ids, x.clone(), loss.item()))
            attempts.append(attempt)
            if len(attempts) <= failures:
                next(model.parameters()).grad.flatten()[0] = float("inf")
            return 1.

        def overflow(event):
            self.assertEqual(len(opt.state), 0)
            self.assertEqual(sched.last_epoch, 0)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, initial[name]))
            events.append(event)

        try:
            _, _, retries = backward_with_retry(compute, opt, scaler, stream,
                                                 model.parameters(), overflow, max_retries)
        except FloatingPointError:
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            return None, events, attempts
        scaler.step(opt)
        scaler.update()
        sched.step()
        self.assertEqual(sched.last_epoch, 1)
        return (copy.deepcopy(model.state_dict()), stream.state_dict(), retries), events, attempts

    def test_retry_replays_rng_batches_and_matches_lower_scale_baseline(self):
        baseline, _, _ = self.run_window(128, 0)
        actual, events, attempts = self.run_window(256, 1)
        self.assertEqual(actual[2], 1)
        self.assertEqual(events[0]["next_amp_scale"], 128)
        for a, b in zip(attempts[0], attempts[1]):
            self.assertEqual(a[0], b[0])
            self.assertTrue(torch.equal(a[1], b[1]))
            self.assertEqual(a[2], b[2])
        for name in actual[0]:
            self.assertTrue(torch.equal(actual[0][name], baseline[0][name]))
        self.assertEqual(actual[1]["cursor"], baseline[1]["cursor"])

    def test_persistent_overflow_stops_without_optimizer_or_scheduler_update(self):
        result, events, attempts = self.run_window(256, 99, max_retries=2)
        self.assertIsNone(result)
        self.assertEqual(len(attempts), 3)
        self.assertFalse(events[-1]["retry"])


if __name__ == "__main__":
    unittest.main()
