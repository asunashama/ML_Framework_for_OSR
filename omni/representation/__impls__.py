import numpy as np
from typing import TypeAlias

array: TypeAlias = np.ndarray


# High-Order Cumulants (HOC)
# ==========================================================================================================
# M20 = np.mean(np.power(signal, 2))
# M21 = np.mean(signal * np.conj(signal))
# M40 = np.mean(np.power(signal, 4))
# M41 = np.mean(np.power(signal, 3) * np.conj(signal))
# M42 = np.mean(np.power(signal, 2) * np.power(np.conj(signal), 2))
# M60 = np.mean(np.power(signal, 6))
# M63 = np.mean(np.power(signal, 3) * np.power(np.conj(signal), 3))
# M80 = np.mean(np.power(signal, 8))
#
# C20 = M20
# C21 = M21
# C40 = M40 - 3 * (M20 ** 2)
# C41 = M41 - 3 * M20 * M21
# C42 = M42 - np.abs(M20) ** 2 - 2 * (M21 ** 2)
# C60 = M60 - 15 * M40 * M20 + 30 * (M20 ** 3)
# C63 = M63 - 6 * M41 * M20 - 9 * M42 * M21 + 18 * M21 * (M20 ** 2) + 12 * (M21 ** 3)
# C80 = M80 - 35 * (M40 ** 2) - 28 * M60 * M20 + 420 * M40 * (M20 ** 2) - 630 * (M20 ** 4)
#
# C20 = np.abs(C20)
# C21 = np.abs(C21)
# C40 = np.abs(C40)
# C41 = np.abs(C41)
# C42 = np.abs(C42)
# C60 = np.abs(C60)
# C63 = np.abs(C63)
# C80 = np.abs(C80)


# : high-order cumulants
# ==========================================================================================================
def cum(signal: array) -> array:
    M20 = np.mean(np.power(signal, 2))
    M21 = np.mean(signal * np.conj(signal))
    M40 = np.mean(np.power(signal, 4))
    M41 = np.mean(np.power(signal, 3) * np.conj(signal))
    M42 = np.mean(np.power(signal, 2) * np.power(np.conj(signal), 2))
    M63 = np.mean(np.power(signal, 3) * np.power(np.conj(signal), 3))

    C40 = M40 - 3 * (M20 ** 2)
    C42 = M42 - np.abs(M20) ** 2 - 2 * (M21 ** 2)
    C63 = M63 - 6 * M41 * M20 - 9 * M42 * M21 + 18 * M21 * (M20 ** 2) + 12 * (M21 ** 3)

    C40 = np.abs(C40)
    C42 = np.abs(C42)
    C63 = np.abs(C63)

    return np.asarray([C40, C40 / C42, C63])


# ... Spectrum
# ==========================================================================================================
class Spectrum(object):

    def __init__(self, d: float, n: int):
        self._n = n
        self._d = d
        self._w = d / n
        self._lb = self._w / 2 - d / 2
        self._ub = d / 2 - self._w / 2

    def __call__(self, signal: array) -> array:
        x = np.asarray(np.clip(signal.real, self._lb, self._ub))
        y = np.asarray(np.clip(signal.imag, self._lb, self._ub))
        i = np.floor((x + self._d / 2) / self._w)
        j = np.floor((y + self._d / 2) / self._w)
        mapping = np.zeros((len(signal), self._n * self._n), dtype=int)
        mapping[np.arange(len(signal)), (j * self._n + i).astype(int).flatten()] = 1
        spectrum = mapping.sum(axis=0).reshape((self._n, self._n))
        spectrum = spectrum / len(signal)
        return spectrum


# ... Representation
# ==========================================================================================================
class Representation(object):
    """convert complex-valued baseband signals (batch) into specified signal representation"""

    def __init__(self, name: str = 'I/Q', **kwargs):
        self._name = name
        self._kwargs = kwargs

    def __call__(self, signals: array, *, mode: str = 'batch') -> array:
        if mode == 'batch':
            match self._name:
                case 'I/Q':
                    return np.asarray([signals.real, signals.imag]).transpose((1, 2, 0))
                case 'A/P':
                    return np.asarray([np.abs(signals), np.angle(signals)]).transpose((1, 2, 0))
                case 'DFT':
                    _cache = np.fft.fft(signals)
                    return np.asarray([_cache.real, _cache.imag]).transpose((1, 2, 0))
                case 'Spectrum':
                    _d, _n = self._kwargs['d'], self._kwargs['n']
                    _transform = Spectrum(_d, _n)
                    _cache = np.zeros((len(signals), _n, _n))
                    for k, s in enumerate(signals):
                        _cache[k] = _transform(s)
                    return _cache
                case 'HOC':
                    _cache = np.zeros((len(signals), 3))
                    for k, s in enumerate(signals):
                        _cache[k] = cum(s)
                    return _cache
                case _:
                    raise ValueError('must specify a correct signal representation')
        elif mode == 'single':
            signal = signals
            match self._name:
                case 'I/Q':
                    return np.asarray([signal.real, signal.imag]).transpose()
                case 'A/P':
                    return np.asarray([np.abs(signal), np.angle(signal)]).transpose()
                case 'FFT':
                    _cache = np.fft.fft(signal)
                    return np.asarray([_cache.real, _cache.imag]).transpose()
                case 'Spectrum':
                    _transform = Spectrum(self._kwargs['d'], self._kwargs['n'])
                    return _transform(signal)
                case 'HOC':
                    return cum(signal)
                case _:
                    raise ValueError('must specify a correct signal representation')
        else:
            raise ValueError('must specify a correct mode (single/batch?)')

    pass
