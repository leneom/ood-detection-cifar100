"""CPU-only torch check that exported affine heads match locally frozen checkpoints."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
import torch
import numpy as np
import hashlib
from geometry_io import ROOT, RUN, read, load, digest, write, require, now


def main():
    require(not torch.cuda.is_available(), 'head audit must run with CUDA masked')
    records = []
    for seed in [0, 1, 2]:
        original = read(ROOT / f'runs/cifar100-seed{seed}/eval-final/results.json')
        require(digest(original['checkpoint']) == original['checkpoint_sha256'], 'checkpoint changed')
        checkpoint = torch.load(original['checkpoint'], map_location='cpu', weights_only=True)
        bank = load(RUN / f'seed{seed}/training-bank.npz')
        for key, tensor in [('weight', checkpoint['model']['fc.weight']), ('bias', checkpoint['model']['fc.bias'])]:
            require(np.array_equal(bank[key], tensor.numpy()), 'exported head is not the frozen classifier head')
        records.append({'seed': seed, 'bank_sha256': digest(RUN / f'seed{seed}/training-bank.npz'),
                        'checkpoint_sha256': original['checkpoint_sha256'],
                        'weight_sha256': hashlib.sha256(bank['weight'].tobytes()).hexdigest(),
                        'bias_sha256': hashlib.sha256(bank['bias'].tobytes()).hexdigest()})
    result = {'state': 'passed', 'audited_at': now(), 'per_seed': records, 'device': 'CPU', 'source_sha256': digest(__file__)}
    write(RUN / 'head-audit.json', result)
    print('all three frozen classifier heads exactly match exported banks', flush=True)


if __name__ == '__main__':
    main()
