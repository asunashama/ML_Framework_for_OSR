import matplotlib.pyplot as plot

from matplotlib.pyplot import Figure, Axes
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullLocator

from omni import *
from omni.utils import Singleton

import seaborn


# ... Color
# ==========================================================================================================
class Color(object, metaclass=Singleton):

    def __call__(self, red: int, green: int, blue: int) -> Tuple[float, float, float]:
        return red / 255, green / 255, blue / 255


RGB = (_Color_ := Color())


# ... Palette
# ==========================================================================================================
class Palette(object, metaclass=Singleton):

    def __call__(self):
        return seaborn.hls_palette(10, l=0.6, s=0.6)

    @staticmethod
    def show(palette: List[Tuple[float, float, float]], *,
             size: float = 1.0, alpha: float = 0.75):
        n = len(palette)

        w = n * size * alpha
        h = size

        figure: Figure = plot.figure(figsize=(w, h))

        axes: Axes = figure.add_subplot()

        axes.imshow(np.arange(n).reshape(1, n), cmap=ListedColormap(list(palette)),
                    interpolation='nearest', aspect='equal')

        axes.set_xticks(np.arange(n) - 0.5)
        axes.set_yticks([-0.5, 0.5])

        axes.spines['top'].set(color=RGB(38, 38, 38), linewidth=1.1)
        axes.spines['bottom'].set(color=RGB(38, 38, 38), linewidth=1.1)
        axes.spines['left'].set(color=RGB(38, 38, 38), linewidth=1.1)
        axes.spines['right'].set(color=RGB(38, 38, 38), linewidth=1.1)

        axes.set_xticklabels(['' for _ in range(n)])

        axes.yaxis.set_major_locator(NullLocator())

        axes.grid(True, axis='x', color=RGB(38, 38, 38), linewidth=1.1)

        plot.get_current_fig_manager().set_window_title('Palette')
        plot.tight_layout()
        plot.show()


palette = (_Palette_ := Palette())
