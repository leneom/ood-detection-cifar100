from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .data import load_cifar, stratified_split, index_hash, synthetic
from .datasets import load_ood, OOD_NAMES, description
from .features import extract, KNN, Mahalanobis
from .metrics import detection_metrics
from .models import build_model
from .runtime import seed_all, environment, atomic_json, file_hash
from .scores import METHODS, score_logits


def image_fingerprint(images):
    """Hash exactly the loaded raw NHWC uint8 input, before normalization."""
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError('input fingerprint requires NHWC uint8 RGB pixels')
    array = images.detach().cpu().contiguous().numpy()
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'layout': 'NHWC',
            'sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest()}


def main():
    p = argparse.ArgumentParser(description='Evaluate a frozen ID-trained classifier')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ood', nargs='+', choices=OOD_NAMES, default=['cifar10'])
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--feature-methods', nargs='*', choices=['knn', 'mahalanobis'], default=[])
    p.add_argument('--knn-k', type=int, default=50)
    p.add_argument('--shrinkage', type=float, default=0.01)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--download', action='store_true')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', default='auto')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('evaluation directory must be empty; choose a new one')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    seed_all(0)
    checkpoint_hash = file_hash(args.checkpoint)
    source_dir = Path(__file__).parent
    source_hashes = {file.name: file_hash(file) for file in source_dir.glob('*.py')}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = build_model(config['model'], config['classes']).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint['model'])
    if config['dataset'] == 'synthetic':
        (train_images, train_labels), (id_images, id_labels) = synthetic(seed=config['seed'])
    else:
        train_images, train_labels = load_cifar(config['data_root'], 'cifar100', True)
        id_images, id_labels = load_cifar(config['data_root'], 'cifar100', False)
    train_indices, val_indices = stratified_split(train_labels.numpy(), config['effective_val_per_class'], config['split_seed'])
    for name, indices in [('train', train_indices), ('val', val_indices)]:
        if index_hash(indices) != checkpoint['split'][f'{name}_indices_sha256']:
            raise ValueError('data split does not match the classifier checkpoint')
    input_pixels = {'id_train': image_fingerprint(train_images[train_indices]),
                    'id_validation': image_fingerprint(train_images[val_indices]),
                    'id_test': image_fingerprint(id_images), 'ood': {}}
    id_logits, id_features = extract(model, id_images, args.batch_size, device)
    val_logits, val_features = extract(model, train_images[val_indices], args.batch_size, device)
    id_accuracy = float((id_logits.argmax(1) == id_labels).float().mean())
    detectors = {}
    feature_fit = None
    if args.feature_methods:
        _, train_features = extract(model, train_images[train_indices], args.batch_size, device)
        if 'knn' in args.feature_methods:
            detectors['knn'] = KNN(args.knn_k, device=device).fit(train_features)
        if 'mahalanobis' in args.feature_methods:
            detectors['mahalanobis'] = Mahalanobis(args.shrinkage, device=device).fit(train_features, train_labels[train_indices])
        feature_fit = {'population': 'ID training split only', 'count': len(train_indices),
                       'indices_sha256': index_hash(train_indices), 'dimensions': train_features.shape[1],
                       'knn_k': args.knn_k, 'mahalanobis_shrinkage': args.shrinkage,
                       'knn': 'L2-normalized kth squared Euclidean distance',
                       'mahalanobis': 'final-layer pooled covariance; fixed isotropic shrinkage; no input perturbation'}
    output.mkdir(parents=True, exist_ok=True)
    result = {
        'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_epoch': checkpoint['epoch'], 'id_dataset': config['dataset'], 'id_test_accuracy': id_accuracy,
        'training_config': config, 'temperature': args.temperature, 'score_direction': 'higher_is_ood',
        'checkpoint_selection': 'ID validation accuracy only', 'feature_fit': feature_fit,
        'benchmark': 'independent CIFAR protocol; not an OpenOOD reproduction',
        'environment': environment(), 'evaluation_source_hashes': source_hashes,
        'input_pixels': input_pixels, 'input_pixel_stage': 'raw uint8 before fixed CIFAR-100 normalization', 'datasets': {},
    }
    snapshot = output / 'evaluation_source'
    snapshot.mkdir()
    for name in source_hashes:
        shutil.copyfile(source_dir / name, snapshot / name)
    atomic_json(output / 'evaluation_source_hashes.json', source_hashes)
    np.savez_compressed(output / 'id_logits.npz', logits=id_logits.numpy(), labels=id_labels.numpy())
    np.savez_compressed(output / 'validation_logits.npz', logits=val_logits.numpy())
    np.savez_compressed(output / 'features.npz', id=id_features.numpy(), validation=val_features.numpy())
    inside = {method: score_logits(id_logits, method, args.temperature).numpy() for method in args.methods}
    validation = {method: score_logits(val_logits, method, args.temperature).numpy() for method in args.methods}
    for method, detector in detectors.items():
        inside[method] = detector.score(id_features).numpy()
        validation[method] = detector.score(val_features).numpy()
    for name in args.ood:
        images = load_ood(config['data_root'], name, len(id_images), args.download)
        input_pixels['ood'][name] = image_fingerprint(images)
        ood_logits, ood_features = extract(model, images, args.batch_size, device)
        np.savez_compressed(output / f'{name}_logits.npz', logits=ood_logits.numpy())
        np.savez_compressed(output / f'{name}_features.npz', features=ood_features.numpy())
        outside = {method: score_logits(ood_logits, method, args.temperature).numpy() for method in args.methods}
        for method, detector in detectors.items():
            outside[method] = detector.score(ood_features).numpy()
        rows, saved_scores = {}, {}
        for method in inside:
            rows[method] = detection_metrics(inside[method], outside[method], validation[method])
            saved_scores[f'{method}_id'] = inside[method]
            saved_scores[f'{method}_ood'] = outside[method]
            saved_scores[f'{method}_val'] = validation[method]
        np.savez_compressed(output / f'{name}_scores.npz', **saved_scores)
        result['datasets'][name] = {'kind': description(name), 'methods': rows}
        print(json.dumps({'ood': name, 'id_test_accuracy': id_accuracy, 'methods': rows}), flush=True)
    if file_hash(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('checkpoint changed during evaluation; evaluate an immutable snapshot')
    if any(file_hash(source_dir / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError('evaluation source changed while running; repeat with stable source')
    atomic_json(output / 'results.json', result)
    return result
