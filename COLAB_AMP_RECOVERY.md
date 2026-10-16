# E3 overflow recovery (step 216)

Observed on T4: finite loss, nonfinite gradients in four ViT LoRA tensors at
step 216, AMP scale 1024. Resume ZIP at step 200 was restored and verified by
the student. This establishes checkpoint integrity, not successful GPU recovery.

`src/amp_retry.py` now retries a complete gradient accumulation window up to
six times with GradScaler backoff. Batch cursor and Python/Torch/NumPy/CUDA RNG
are replayed. Failed attempts do not update weights, optimizer or scheduler.
Loss NaN/Inf still stops immediately. Persistent gradient overflow still stops.
LR, batch size, seed, target steps and checkpoint configuration are unchanged.
Retries are recorded in `amp_retry.jsonl` and successful-step `amp_retries`.
This is a numerical recovery policy change, not bitwise equivalence to the
previous fail-fast implementation; disclose it when reporting the new run.

CPU checks: `python -m unittest src.test_amp_retry src.test_resume -v`.
No pretrained models are loaded locally. Real T4 recovery is still pending.

For an already open Colab workspace, upload only these three updated files:
`src/train.py`, `src/amp_retry.py`, `src/test_amp_retry.py`. Copy uploaded files
into `/content/vlm_lab300/src/` and run the two CPU retry fixtures there.
Do not extract the complete bundle over an existing workspace.

Then resume the latest verified checkpoint with the existing E3_ARGS and smoke
marker, adding `--pause-at 220 --checkpoint-every 10`. This is a short real-data
recovery validation through the failing window, not a new full run or a change
to the 300-step LR horizon. Require `step-0220` and BACKUP ZIP OK before proceeding.
After that rerun the regular full E3 cell without pause-at; it resumes from 220.
The previously passed eight-step smoke need not be rerun unchanged. Existing
submitted evidence and old checkpoints must not be edited.
