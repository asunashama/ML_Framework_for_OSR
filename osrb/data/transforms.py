import numpy as np


def complex_to_iq(signals: np.ndarray) -> np.ndarray:
    """Convert complex-valued baseband signals (N, L) into float32 I/Q array (N, L, 2)."""
    signals = np.asarray(signals)
    if np.iscomplexobj(signals):
        return np.stack([signals.real, signals.imag], axis=-1).astype(np.float32)
    if signals.ndim == 3 and signals.shape[-1] == 2:
        return signals.astype(np.float32)
    if signals.ndim == 3 and signals.shape[1] == 2:
        return np.transpose(signals, (0, 2, 1)).astype(np.float32)
    raise ValueError(f"Unsupported signal shape {signals.shape} and dtype {signals.dtype}")


def rms_normalize(signals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize per-sample RMS power to 1."""
    signals = np.asarray(signals)
    if np.iscomplexobj(signals):
        pwr = np.sqrt(np.mean(np.abs(signals) ** 2, axis=-1, keepdims=True)) + eps
        return (signals / pwr).astype(np.complex64)
    pwr = np.sqrt(np.mean(np.sum(signals ** 2, axis=-1, keepdims=True), axis=1, keepdims=True)) + eps
    return (signals / pwr).astype(np.float32)


def remove_dc_offset(signals: np.ndarray) -> np.ndarray:
    """Remove per-sample DC offset (mean value)."""
    signals = np.asarray(signals)
    mean_val = np.mean(signals, axis=1, keepdims=True)
    return signals - mean_val


def compensate_coarse_cfo(signals: np.ndarray) -> np.ndarray:
    """Estimate and compensate per-frame average phase increment (coarse CFO)."""
    signals = np.asarray(signals)
    if not np.iscomplexobj(signals):
        c = signals[..., 0] + 1j * signals[..., 1]
    else:
        c = signals
    # Lag-1 autocorrelation phase estimate
    prod = np.sum(c[:, 1:] * np.conj(c[:, :-1]), axis=-1, keepdims=True)
    dphi = np.angle(prod)
    n = np.arange(c.shape[1], dtype=np.float32)[None, :]
    c_comp = (c * np.exp(-1j * dphi * n)).astype(np.complex64)
    if not np.iscomplexobj(signals):
        return complex_to_iq(c_comp)
    return c_comp


def preprocess_signals(
    signals: np.ndarray,
    remove_dc: bool = False,
    compensate_cfo: bool = False,
    normalize_rms: bool = False,
) -> np.ndarray:
    """Apply optional physical-layer preprocessing to complex baseband signals."""
    out = np.asarray(signals)
    if remove_dc:
        out = remove_dc_offset(out)
    if compensate_cfo:
        out = compensate_coarse_cfo(out)
    if normalize_rms:
        out = rms_normalize(out)
    return out
