"""Independent training-only fitted-state audit before any geometry OOD evaluation."""
import hashlib
import json
import statistics
import numpy as np
import scipy.linalg
from geometry_io import ROOT, RUN, read, write, load, digest, require, now


def main():
    require(read(RUN / 'fit-status.json')['state'] == 'complete', 'geometry fitting incomplete')
    manifest = read(RUN / 'fit-inputs.json')
    for name, expected in manifest['original_artifacts'].items():
        require(digest(ROOT / name) == expected, 'original fit input changed')
    for name, expected in manifest['source_hashes'].items():
        require(digest(ROOT / 'tools' / name) == digest(RUN / 'fit-source' / name) == expected, 'fit source changed')
    records = []
    for seed in [0, 1, 2]:
        directory = RUN / f'seed{seed}'
        fit, state, bank = read(directory / 'fit.json'), load(directory / 'fit-state.npz'), load(directory / 'training-bank.npz')
        require(fit['state'] == 'complete' and fit['fit_inputs_sha256'] == digest(RUN / 'fit-inputs.json'), 'fit inputs changed')
        require(fit['fit_state_sha256'] == digest(directory / 'fit-state.npz'), 'fitted state changed')
        require(fit['training_bank_sha256'] == digest(directory / 'training-bank.npz'), 'bank changed after fit')
        require(fit['fitted_at'] > manifest['prepared_at'], 'fit predates input freezing')
        require(fit['training_count'] == 45000 and fit['principal_dimensions'] == 256 and fit['react_global_quantile'] == .99,
                'fitting settings changed')
        require(fit['sample_mean_subtracted'] is False and fit['normalization'] == 'unnormalized final pooled features', 'wrong centering/normalization')
        f, w, b = bank['features'].astype(np.float64), bank['weight'].astype(np.float64), bank['bias'].astype(np.float64)
        # Minimum-norm affine-origin solution uses a separate least-squares solver.
        origin = scipy.linalg.lstsq(w, -b, cond=1e-15, lapack_driver='gelsd')[0]
        require(np.allclose(state['origin'], origin, atol=1e-12, rtol=1e-12), 'origin does not match independent least squares')
        # Accumulate training second moment in independent blocks, with no sample-mean subtraction.
        moment = np.zeros((512, 512), dtype=np.float64)
        for start in range(0, len(f), 1024):
            x = f[start:start + 1024] - origin
            moment += x.T @ x
        moment /= len(f)
        require(np.allclose(moment, state['second_moment'], atol=1e-12, rtol=1e-12), 'training second moment mismatch')
        values, vectors = scipy.linalg.eigh(moment, driver='evr')
        require(np.allclose(values[::-1], state['eigenvalues_descending'], atol=1e-11, rtol=1e-10), 'eigenspectrum mismatch')
        principal, residual = state['principal_basis'], state['residual_basis']
        require(np.allclose(principal.T @ principal, np.eye(256), atol=1e-12), 'principal basis not orthonormal')
        require(np.allclose(residual.T @ residual, np.eye(256), atol=1e-12), 'residual basis not orthonormal')
        require(np.allclose(principal.T @ residual, np.zeros((256, 256)), atol=1e-12), 'subspaces overlap')
        reference_projector = vectors[:, :256] @ vectors[:, :256].T
        fitted_projector = residual @ residual.T
        projector_error = float(np.max(np.abs(reference_projector - fitted_projector)))
        require(projector_error < 1e-9, 'independent residual subspace differs')
        x = f - origin
        # Complement-of-principal norm independently verifies every saved training residual.
        squared = np.sum(x*x, axis=1) - np.sum((x @ principal)**2, axis=1)
        require(squared.min() >= -1e-9, 'invalid residual norm identity')
        residual_reference = np.sqrt(np.maximum(squared, 0))
        require(np.allclose(residual_reference, state['training_residual'], atol=1e-9, rtol=1e-9), 'training residual vector mismatch')
        affine_reference = f @ w.T + b
        train_max = affine_reference.max(axis=1)
        require(np.allclose(train_max, state['training_max_affine_logit'], atol=1e-12, rtol=1e-12), 'training affine maxima mismatch')
        # Use independently reconstructed residuals and Python fmean, not production NumPy means.
        alpha = statistics.fmean(train_max.tolist()) / statistics.fmean(residual_reference.tolist())
        require(abs(alpha - fit['alpha']) < 1e-8 and alpha > 0, 'train-only alpha mismatch')
        ordered = np.sort(f.reshape(-1))
        position = .99 * (len(ordered) - 1)
        lower, upper = int(np.floor(position)), int(np.ceil(position))
        clipping = float(ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower]))
        require(clipping == fit['clipping_level'], 'global linear-interpolated training quantile mismatch')
        records.append({'seed': seed, 'fit_json_sha256': digest(directory / 'fit.json'),
                        'fit_state_sha256': digest(directory / 'fit-state.npz'), 'training_rows': len(f),
                        'alpha': fit['alpha'], 'independent_alpha_error': abs(alpha - fit['alpha']),
                        'clipping_level': clipping, 'principal_dimensions': 256,
                        'second_moment_max_error': float(np.max(np.abs(moment - state['second_moment']))),
                        'residual_projector_max_error': projector_error,
                        'training_residual_max_error': float(np.max(np.abs(residual_reference - state['training_residual'])))})
    result = {'state': 'passed', 'audited_at': now(), 'per_seed': records,
              'original_artifact_hash_checks': len(manifest['original_artifacts']),
              'training_residual_rows_checked': 135000, 'global_quantile_coordinates_per_seed': 45000 * 512,
              'fit_inputs_sha256': digest(RUN / 'fit-inputs.json'), 'source_sha256': digest(__file__),
              'boundary': 'all fitting independently checked on training ID only; no geometry OOD scoring occurred'}
    write(RUN / 'fit-audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
