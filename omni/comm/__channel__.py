from .__omni__ import *


# ... SamplingFrequencyDrift
# =====================================================================================================
class SamplingFrequencyDrift(object):

    def __init__(self):
        pass


# ... CarrierFrequencyOffset
# =====================================================================================================
class CarrierFrequencyOffset(object):

    def __init__(self,
                 sigma: float = 0.0001,
                 frequency: float = 2.4E9,
                 duration: float = 1 / 100E6):
        self._sigma = sigma
        self._frequency = frequency
        self._duration = duration

    def __call__(self, signal: array) -> array:
        offset = np.random.normal() * self._sigma * self._frequency
        return signal * np.exp(1j * 2 * np.pi * offset * np.arange(len(signal)) * self._duration)


# ... Rayleigh
# =====================================================================================================
class Rayleigh(object):
    """Rayleigh Fading Channel"""

    def __init__(self, sigma: float = math.sqrt(0.5)):
        self._sigma = sigma

    def __call__(self, signal: array) -> array:
        factor = np.random.normal() + 1j * np.random.normal()
        return signal * factor * self._sigma

    def __str__(self):
        return 'Rayleigh'


# ... AWGN
# =====================================================================================================
class AWGN(object):
    """Additive White Gaussian Noise"""

    def __init__(self, SNR: float, *,
                 power: Optional[float] = None):
        self._power = power
        self._SNR = SNR

    def __call__(self, signal: array) -> array:
        if self._power is None:
            Ps = np.mean(np.abs(signal) ** 2)
        else:
            Ps = self._power
        Pn = Ps / np.power(10, self._SNR / 10)
        if signal.dtype == complex:
            real = np.random.randn(len(signal))
            imag = np.random.randn(len(signal))
            noise = (real + 1j * imag) * np.sqrt(Pn / 2)
        else:
            noise = np.random.randn(len(signal)) * np.sqrt(Pn)
        return signal + noise

    def __str__(self):
        return 'AWGN'

    @property
    def SNR(self) -> float:
        return self._SNR


# ... ColouredNoise
# =====================================================================================================
class ColouredNoise(object):

    def __init__(self, SNR: float):
        self._SNR = SNR

    @staticmethod
    def colored(data: np.ndarray) -> np.ndarray:
        output = np.zeros_like(data)
        data = np.pad(data, (1, 0), 'edge')
        for k in range(len(output)):
            output[k] = data[k + 1] + 0.5 * data[k]
        return output

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        Ps = np.mean(np.abs(signal) ** 2)
        Pn = Ps / np.power(10, self._SNR / 10)
        if signal.dtype == complex:
            real = np.random.randn(len(signal))
            imag = np.random.randn(len(signal))
            noise = real + 1j * imag
            output = self.colored(noise)
            _Pn = np.mean(np.abs(noise) ** 2)
            Po = np.mean(np.abs(output) ** 2)
            output = output * np.sqrt(_Pn / Po)
            noise = output * np.sqrt(Pn / 2)
        else:
            raise TypeError
        return signal + noise

    def __str__(self):
        return 'Colored Noise'

    @property
    def SNR(self) -> float:
        return self._SNR
