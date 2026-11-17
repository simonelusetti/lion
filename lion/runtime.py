import torch


def device_ids(cfg):
    if not cfg.runtime.device.startswith("cuda"):
        return []
    first = torch.device(cfg.runtime.device).index or 0
    return list(cfg.runtime.devices) if cfg.runtime.devices is not None else [
        first, *[i for i in range(torch.cuda.device_count()) if i != first]]


def parallel(module, cfg):
    ids = device_ids(cfg)
    return torch.nn.DataParallel(module, device_ids=ids) if cfg.runtime.parallel and len(ids) > 1 else module


def configure(cfg):
    torch.set_num_threads(cfg.runtime.threads)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.runtime.tf32
    torch.backends.cudnn.allow_tf32 = cfg.runtime.tf32


def autocast(cfg):
    return torch.autocast(device_type=torch.device(cfg.runtime.device).type,
                          dtype=torch.float16 if cfg.runtime.amp == "fp16" else torch.bfloat16,
                          enabled=cfg.runtime.amp in ("bf16", "fp16"))
