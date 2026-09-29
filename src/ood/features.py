"""Training-ID feature baselines. Hyperparameters are fixed before test use.

KNN is normalized kth squared Euclidean distance (k=50 by default).
Mahalanobis is the nearest class distance under a pooled, regularized
covariance of final-layer features. No perturbation or multi-layer regression.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .data import preprocess


@torch.inference_mode()
def extract(model, images, batch_size=256, device="cpu"):
    model.eval()
    layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    if not layers:
        raise ValueError("model has no final linear classifier")
    captured, features, logits = [], [], []
    def hook(module, inputs):
        captured.append(inputs[0].detach())
    handle = layers[-1].register_forward_pre_hook(hook)
    try:
        for start in range(0, len(images), batch_size):
            captured.clear()
            output = model(preprocess(images[start:start + batch_size].to(device)))
            feature = captured[0]
            if not torch.isfinite(feature).all() or not torch.isfinite(output).all():
                raise RuntimeError("non-finite feature or logits")
            features.append(feature.cpu())
            logits.append(output.cpu())
    finally:
        handle.remove()
    return torch.cat(logits), torch.cat(features)


def validate_features(features):
    if features.ndim != 2 or not len(features) or not features.shape[1] or not torch.isfinite(features).all():
        raise ValueError("features must be a nonempty finite [N, D] matrix")


class KNN:
    def __init__(self, k=50, chunk_size=256, device="cpu"):
        self.k, self.chunk_size, self.device = k, chunk_size, device

    def fit(self, features):
        validate_features(features)
        if not 1 <= self.k <= len(features) or self.chunk_size <= 0:
            raise ValueError("k and chunk size exceed valid bounds")
        if (features.norm(dim=1) < 1e-12).any():
            raise ValueError("cannot normalize zero training features")
        self.bank = F.normalize(features.float(), dim=1).to(self.device)
        self.bank_norm = self.bank.square().sum(dim=1)
        return self

    @torch.inference_mode()
    def score(self, features):
        validate_features(features)
        if features.shape[1] != self.bank.shape[1] or (features.norm(dim=1) < 1e-12).any():
            raise ValueError("incompatible or zero query features")
        queries = F.normalize(features.float(), dim=1)
        result = []
        for batch in queries.split(self.chunk_size):
            batch = batch.to(self.device)
            distances = (batch.square().sum(dim=1, keepdim=True) + self.bank_norm[None] - 2 * batch @ self.bank.T).clamp_min_(0)
            result.append(distances.topk(self.k, largest=False, dim=1).values[:, -1].cpu())
        return torch.cat(result)


class Mahalanobis:
    def __init__(self, shrinkage=0.01, chunk_size=256, device="cpu"):
        if not 0 < shrinkage <= 1 or chunk_size <= 0:
            raise ValueError("shrinkage must be in (0, 1], chunk size must be positive")
        self.shrinkage, self.chunk_size, self.device = shrinkage, chunk_size, device

    def fit(self, features, labels):
        validate_features(features)
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("labels must match training features")
        features = features.double().cpu()
        labels = labels.cpu()
        classes = labels.unique(sorted=True)
        means = torch.stack([features[labels == label].mean(dim=0) for label in classes])
        residuals = features.clone()
        for i, label in enumerate(classes):
            residuals[labels == label] -= means[i]
        covariance = residuals.T @ residuals / len(features)
        scale = covariance.diag().mean().clamp_min(1e-12)
        covariance = (1 - self.shrinkage) * covariance + self.shrinkage * scale * torch.eye(features.shape[1], dtype=torch.float64)
        self.precision = torch.linalg.inv(covariance).to(self.device)
        self.means = means.to(self.device)
        self.means_precision = self.means @ self.precision
        self.mean_squares = (self.means_precision * self.means).sum(dim=1)
        return self

    @torch.inference_mode()
    def score(self, features):
        validate_features(features)
        if features.shape[1] != self.means.shape[1]:
            raise ValueError("feature dimensions differ")
        result = []
        for batch in features.double().split(self.chunk_size):
            batch = batch.to(self.device)
            distances = (batch @ self.precision * batch).sum(dim=1, keepdim=True) - 2 * batch @ self.means_precision.T + self.mean_squares[None]
            result.append(distances.amin(dim=1).clamp_min(0).cpu())
        return torch.cat(result)
