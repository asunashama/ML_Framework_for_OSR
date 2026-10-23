from .__modulator__ import Modulator
from .__omni__ import *
from .__utils__ import rcosdesign, upfirdn


# ... Radio
# =====================================================================================================
class Radio(object):

    def __init__(self):
        self._modulator: Optional[Modulator] = None
        self._pipeline = []

    def attach(self, component: Callable[[array], array]):
        if str(component) == 'Modulator':
            self._modulator = component
        self._pipeline.append(component)
        return self

    def __call__(self, sequence: array) -> array:
        for component in self._pipeline:
            sequence = component(sequence)
        return sequence

    @property
    def order(self) -> int:
        if self._modulator is not None:
            return self._modulator.order
        raise AttributeError

    @property
    def modulation(self) -> str:
        if self._modulator is not None:
            return self._modulator.name
        raise AttributeError


# ... PulseShaping
# =====================================================================================================
class PulseShaping(object):
    """Baseband Pulse-Shaping Filter"""

    def __init__(self, sps: int = 8, factor: float = 0.35):
        self._h = rcosdesign(factor, 8, sps, 'normal')
        self._sps = sps

    def __call__(self, signal: array) -> array:
        return upfirdn(self._h, signal, self._sps, 1)


class Fingerprints(object):

    def __init__(self, config: Dict[str, Tuple[float, float, float, float]]):
        self._config = config

    @staticmethod
    def _nonlinearity(input: array, coefficients: Tuple[float, float, float, float]) -> array:
        c1, c2, c3, c4 = coefficients
        return (c1 * (input ** c2) + c3 * (input ** (c2 + 1))) / \
            (1 + c4 * (input ** c2))

    def __call__(self, signal: array) -> array:
        output = self._nonlinearity(np.abs(signal), self._config['amp']) * np.exp(
            1j * (np.angle(signal) + self._nonlinearity(np.abs(signal), self._config['phi'])))
        return output

    def __str__(self):
        return 'Fingerprints'


class PowerNormalization(object):

    def __init__(self, factor: float = 1.0):
        self._factor = factor
        pass

    def __call__(self, signal: array) -> array:
        return signal * np.sqrt(self._factor / np.mean(np.abs(signal) ** 2))

    def __str__(self):
        return 'Power Normalization'
