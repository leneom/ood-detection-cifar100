from __future__ import annotations

import math
import torch


METHODS = ("msp", "energy", "maxlogit", "entropy")


def score_logits(logits, method, temperature=1.0):
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [N, K], with N>0 and K>=2")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = logits.float()
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite logits")
    if method == "msp":
        return 1.0 - torch.softmax(logits / temperature, dim=1).amax(dim=1)
    if method == "energy":
        return -temperature * torch.logsumexp(logits / temperature, dim=1)
    if method == "maxlogit":
        return -logits.amax(dim=1)
    if method == "entropy":
        log_p = torch.log_softmax(logits / temperature, dim=1)
        return -(log_p.exp() * log_p).sum(dim=1)
    raise ValueError(f"unknown score method: {method}")
