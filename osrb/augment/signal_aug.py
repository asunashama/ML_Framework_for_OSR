import random
import numpy as np


def _op0(signal: np.ndarray) -> np.ndarray:
    return signal


def _op1(signal: np.ndarray) -> np.ndarray:
    return signal * np.exp(1j * (0.5 * np.pi))


def _op2(signal: np.ndarray) -> np.ndarray:
    return -signal


def _op3(signal: np.ndarray) -> np.ndarray:
    return signal * np.exp(1j * (1.5 * np.pi))


def _op4(signal: np.ndarray) -> np.ndarray:
    return np.conj(signal)


def _op5(signal: np.ndarray) -> np.ndarray:
    return np.conj(-signal)


_OPS_QUADRANT = [_op0, _op1, _op2, _op3]
_OPS_FLIP = [_op0, _op2, _op4, _op5]


def permute_signal(
    signal: np.ndarray,
    num_of_segments: int = 2,
    phase_mode: str = "quadrant",
) -> np.ndarray:
    """
    Split a complex baseband signal into K random segments, apply random orthogonal
    phase rotation (or conjugate flip) to each segment, and randomly permute segment order.
    """
    if num_of_segments < 2:
        ops = _OPS_QUADRANT if phase_mode == "quadrant" else _OPS_FLIP
        return ops[random.randint(0, 3)](signal)

    length = len(signal)
    pos = []
    while len(pos) < num_of_segments - 1:
        p = random.randint(0, length)
        if p not in pos:
            pos.append(p)
    pos = [0] + sorted(pos) + [length]

    segments = [signal[pos[k] : pos[k + 1]] for k in range(num_of_segments)]
    index = np.random.permutation(num_of_segments)
    ops = _OPS_QUADRANT if phase_mode == "quadrant" else _OPS_FLIP
    transformed = [ops[random.randint(0, 3)](segments[k]) for k in index]
    return np.concatenate(transformed)


def augment_on_batch(
    signals: np.ndarray,
    num_of_segments: int = 2,
    phase_mode: str = "quadrant",
    concat_original: bool = False,
) -> np.ndarray:
    """
    Apply segment permutation + orthogonal phase rotation across a batch of complex signals.
    If concat_original=True (as in train_augment_100.py), returns [original; augmented] of shape (2N, L).
    """
    aug = np.zeros_like(signals)
    for k, sig in enumerate(signals):
        aug[k] = permute_signal(sig, num_of_segments=num_of_segments, phase_mode=phase_mode)
    if concat_original:
        return np.concatenate([signals, aug], axis=0)
    return aug


def soften_labels(targets: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """Apply label smoothing to one-hot target labels."""
    if alpha <= 0.0:
        return targets.astype(np.float32)
    num_classes = targets.shape[-1]
    return (targets * (1.0 - alpha) + (alpha / num_classes) * np.ones_like(targets)).astype(np.float32)
