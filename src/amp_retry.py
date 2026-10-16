"""Retry an overflowed accumulation window, never skip an optimizer step."""
import copy
import math

from .resume import capture_rng, restore_rng


def backward_with_retry(compute_backward, optimizer, scaler, stream, params,
                        on_overflow, max_retries=6):
    """Callback computes/backprops one full window; only caller steps optimizer/LR.

    Replay batch cursor and dropout/augmentation RNG, but retain the lower scale.
    No optimizer/model parameters are updated during failed attempts.
    """
    import torch
    params = list(params)
    stream_before = copy.deepcopy(stream.state_dict())
    rng_before = capture_rng(include_cuda=any(p.is_cuda for p in params))
    for attempt in range(max_retries + 1):
        if attempt:
            stream.load_state_dict(stream_before)
            restore_rng(rng_before)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_backward()
        scaler.unscale_(optimizer)
        try:
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        except RuntimeError:
            # Do not disguise unrelated clipping/runtime errors as AMP overflow.
            bad = [i for i, p in enumerate(params)
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
            if not bad:
                grads = [p.grad.detach().float().norm() for p in params if p.grad is not None]
                if not grads or torch.isfinite(torch.stack(grads).norm()):
                    raise
            old_scale = scaler.get_scale()
            new_scale = old_scale * scaler.get_backoff_factor()
            retry = attempt < max_retries and math.isfinite(new_scale) and new_scale > 0
            on_overflow({"attempt": attempt + 1, "loss": loss,
                         "amp_scale": old_scale, "next_amp_scale": new_scale,
                         "retry": retry, "nonfinite_param_indices": bad})
            optimizer.zero_grad(set_to_none=True)
            if not retry:
                raise FloatingPointError("AMP overflow persists; optimizer step NOT applied.")
            # Explicit update resets GradScaler's unscale stage, without optimizer.step.
            scaler.update(new_scale=new_scale)
            continue
        return loss, norm, attempt
    raise AssertionError("Unreachable")
