import torch


def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def is_fp8_supported_gpu():
    if not torch.cuda.is_available():
        return False
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    return (major == 8 and minor == 9) or (major >= 9)


def ceil_div(x, y):
    return (x + y - 1) // y
