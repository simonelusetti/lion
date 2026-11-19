import torch


class FrozenDataParallel(torch.nn.DataParallel):
    """Reuse replicas of an immutable encoder across batch forwards."""

    def __init__(self, module, device_ids):
        super().__init__(module, device_ids=device_ids)
        self._frozen_replicas = []

    def replicate(self, module, device_ids):
        if len(device_ids) > len(self._frozen_replicas):
            with torch.inference_mode(False), torch.no_grad():
                self._frozen_replicas = super().replicate(module, device_ids)
        return self._frozen_replicas[:len(device_ids)]

    def _apply(self, fn, recurse=True):
        self._frozen_replicas = []
        return super()._apply(fn, recurse=recurse)


def device_ids(cfg):
    if not cfg.runtime.device.startswith("cuda"):
        return []
    first = torch.device(cfg.runtime.device).index or 0
    return list(cfg.runtime.devices) if cfg.runtime.devices is not None else [
        first, *[i for i in range(torch.cuda.device_count()) if i != first]]


def parallel(module, cfg, *, frozen=False):
    ids = device_ids(cfg)
    if not cfg.runtime.parallel or len(ids) <= 1:
        return module
    wrapper = FrozenDataParallel if frozen else torch.nn.DataParallel
    return wrapper(module, device_ids=ids)


def configure(cfg, *, threads=None):
    torch.set_num_threads(cfg.runtime.threads if threads is None else threads)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.runtime.tf32
    torch.backends.cudnn.allow_tf32 = cfg.runtime.tf32


def autocast(cfg):
    return torch.autocast(device_type=torch.device(cfg.runtime.device).type,
                          dtype=torch.float16 if cfg.runtime.amp == "fp16" else torch.bfloat16,
                          enabled=cfg.runtime.amp in ("bf16", "fp16"))
