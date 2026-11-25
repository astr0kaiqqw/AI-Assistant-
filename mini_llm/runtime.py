import torch


def select_device(requested: str) -> torch.device:
    available = torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise ValueError("MPS is unavailable. Use --device auto or --device cpu.")
    if requested == "auto":
        requested = "mps" if available else "cpu"
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
