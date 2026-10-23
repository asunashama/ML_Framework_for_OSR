from omni import *
from omni.plotting import *


class Histogram(object):

    def __init__(self, samples: array, scope: Tuple[float, float], n: int):
        self._samples = samples

        a, b = scope

        self._a = a
        self._b = b
        self._w = (b - a) / n
        self._n = n

        samples = np.asarray(np.clip(samples, self._a + self._w / 5, self._b - self._w / 5))
        index = np.floor((samples - self._a) / self._w).astype(int)
        mapping = np.zeros((len(samples), self._n), dtype=int)
        mapping[np.arange(len(samples)), index] = 1
        histogram = mapping.sum(axis=0) / len(samples)
        histogram *= 1.0 / self._w

        self._position = self._a + self._w / 2 + self._w * np.arange(self._n)
        self._histogram = histogram

    def show(self, *, export: str = 'export/histogram.pdf'):
        figure, axes = plot.subplots(figsize=(5.2, 3.8))
        axes.set_xlim(self._a - 3 * self._w, self._b + 3 * self._w)
        axes.tick_params(axis='both', width=0.5, colors=RGB(38, 38, 38))
        # axes.grid(True, which='major', axis='y', linewidth=0.5, color=RGB(223, 223, 223))
        axes.hist(self._samples[:, None], bins=self._n, range=(self._a, self._b), density=True,
                  color=RGB(72, 102, 223), edgecolor=RGB(247, 249, 245), linewidth=0.5)
        axes.plot(self._position, self._histogram, color=RGB(239, 52, 52), linewidth=1.0)
        plot.get_current_fig_manager().set_window_title('Histogram')
        plot.tight_layout()
        if export is not None:
            plot.savefig(export, dpi=600, bbox_inches='tight', pad_inches=0.05)
        plot.show()


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

# from sklearn.datasets import make_blobs
# from sklearn.neighbors import KernelDensity

# 1-D Kernel Density Estimation ...
# ==========================================================================================================
# def _1D_how_to_use():
#     """Gaussian Kernel Density Estimation"""
#     data = np.concatenate(
#         [np.random.normal(-3, 1.6, 1000), np.random.normal(3, 1, 1000)]
#     )[:, None]
#     estimation = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(data)
#     x = np.linspace(-10, 10, 1000)[:, None]
#     p = np.exp(estimation.score_samples(x))
#     figure, axes = plot.subplots(figsize=(5.2, 3.8))
#     axes.plot(x, p, color=RGB(162, 5, 63))
#     axes.tick_params(width=0.5, colors=RGB(38, 38, 38))
#     axes.grid(True, linewidth=0.5, color=RGB(223, 223, 223))
#     plot.tight_layout()
#     plot.show()


# 2-D Kernel Density Estimation ...
# ==========================================================================================================
# def _2D_how_to_use(n: int = 128):
#     centers = [[1.5, 1.5], [-1.2, -1.2], [1.3, -1.8]]
#     data, _ = make_blobs(2000, centers=centers, cluster_std=0.4)
#     estimation = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(data)
#     x, y = np.meshgrid(np.linspace(-3.2, 3.2, n), np.linspace(-3.2, 3.2, n))
#     x, y = np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)
#     xy = np.concatenate([x, y], axis=-1).reshape((n * n, 2))
#     p = np.exp(estimation.score_samples(xy)).reshape((n, n))
#     figure, axes = plot.subplots(figsize=(4.2, 4.2))
#     axes.imshow(p, origin="lower", cmap="jet")
#     axes.axis("off")
#     plot.tight_layout()
#     plot.show()


# 2-D Kernel Density Estimation ... (Spectrum)
# ==========================================================================================================
# class KernelSpectrum(object):
#     """Spectrum with Kernel Density Estimation"""
#
#     def __init__(self, d: float, n: int):
#         self._d = d
#         self._n = n
#         x, y = np.meshgrid(np.linspace(-d / 2, d / 2, n), np.linspace(-d / 2, d / 2, n))
#         x, y = np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)
#         self._xy = np.concatenate([x, y], axis=-1).reshape((n * n, 2))
#
#     def __call__(self, signal: array) -> array:
#         data = np.reshape(signal, (len(signal), 1))
#         data = np.concatenate([data.real, data.imag], axis=-1)
#         estimation = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(data)
#         spectrum = np.exp(estimation.score_samples(self._xy))
#         spectrum = np.reshape(spectrum, (self._n, self._n))
#         return spectrum
