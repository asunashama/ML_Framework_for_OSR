import matplotlib.pyplot as plot
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap
from matplotlib.container import BarContainer
from matplotlib.image import AxesImage
from matplotlib.pyplot import Figure, Axes, Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .__palette__ import palette, RGB

import seaborn

__all__ = ['palette', 'RGB', 'Figure', 'Axes', 'Line2D', 'Colorbar', 'BarContainer', 'plot',
           'make_axes_locatable', 'ListedColormap', 'LinearSegmentedColormap',
           'MaxNLocator', 'AxesImage', 'Normalize', 'seaborn']

edge = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe %s'
chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'

plot.rcParams.update(
    {
        'axes.titlesize': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'lines.linestyle': '-',
        'lines.linewidth': '1.5'
    }
)
