"""State phục hồi ở ranh giới bước tối ưu; không tải model khi import."""
import importlib.metadata
import random


class BatchStream:
    """Giữ đúng thứ tự DataLoader shuffle=True,num_workers=0 và không encode lại ảnh đã đọc."""
    def __init__(self, size, batch_size, seed):
        import torch
        if size < 1 or batch_size < 1:
            raise ValueError("Dataset và batch phải dương.")
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.batches, self.cursor, self.epoch = [], 0, -1

    def next_indices(self):
        import torch
        from torch.utils.data import BatchSampler, RandomSampler
        if self.cursor == len(self.batches):
            # DataLoader lấy base_seed trước khi RandomSampler tạo hoán vị.
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
            sampler = RandomSampler(range(self.size), generator=self.generator)
            self.batches = list(BatchSampler(sampler, self.batch_size, drop_last=False))
            self.cursor = 0
            self.epoch += 1
        batch = self.batches[self.cursor]
        self.cursor += 1
        return batch

    def state_dict(self):
        return {"size": self.size, "batch_size": self.batch_size, "epoch": self.epoch,
                "batches": self.batches, "cursor": self.cursor, "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        if (state["size"], state["batch_size"]) != (self.size, self.batch_size):
            raise ValueError("Batch stream khác dataset/batch checkpoint.")
        batches = state["batches"]
        flat = [i for batch in batches for i in batch]
        if batches and sorted(flat) != list(range(self.size)):
            raise ValueError("Hoán vị batch checkpoint không hợp lệ.")
        if not 0 <= state["cursor"] <= len(batches):
            raise ValueError("Cursor checkpoint không hợp lệ.")
        self.batches, self.cursor, self.epoch = batches, state["cursor"], state["epoch"]
        self.generator.set_state(state["generator"])


def capture_rng(include_cuda=False):
    import torch
    state = {"python": random.getstate(), "torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if include_cuda else []}
    try:
        import numpy as np
        name, keys, position, has_gauss, cached = np.random.get_state()
        state["numpy"] = (name, keys.tolist(), position, has_gauss, cached)
    except ImportError:
        pass
    return state


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Số GPU khác checkpoint.")
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        import numpy as np
        name, keys, position, has_gauss, cached = state["numpy"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))


def runtime_versions():
    import torch
    versions = {"torch": str(torch.__version__)}
    for name in ("transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def training_state(optimizer, scheduler, scaler, stream, *, step, fingerprint, trainable_names,
                   smoke, target_steps):
    return {"format_version": 1, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "stream": stream.state_dict(),
            "rng": capture_rng(include_cuda=any(p.is_cuda for g in optimizer.param_groups for p in g["params"])),
            "step": step,
            "fingerprint": fingerprint, "trainable_names": trainable_names,
            "smoke": smoke, "target_steps": target_steps, "versions": runtime_versions()}


def restore_training_state(state, optimizer, scheduler, scaler, stream, *, fingerprint,
                           trainable_names, smoke, target_steps):
    if state.get("format_version") != 1 or state["fingerprint"] != fingerprint:
        raise ValueError("Resume state không cùng format/config/dataset.")
    if state["trainable_names"] != trainable_names:
        raise ValueError("Thứ tự tham số optimizer thay đổi.")
    if state["smoke"] != smoke or state["target_steps"] != target_steps:
        raise ValueError("Không resume smoke thành full hoặc đổi số bước đích.")
    if state["versions"] != runtime_versions():
        raise ValueError("Phiên bản runtime khác checkpoint; không resume chính xác qua phiên bản.")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    stream.load_state_dict(state["stream"])
    # Gọi cuối cùng, sau khi model/optimizer đã được tạo và nạp.
    restore_rng(state["rng"])
    return state["step"]
