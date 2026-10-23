import matplotlib.pyplot as plot

from omni import *
from omni.utils import Singleton


# ... Palette
# ===============================================================================================================
class Palette(object, metaclass=Singleton):
    _color = {
        'Red': (255, 0, 0),
        'Crimson': (220, 20, 60),
        'Magenta': (255, 0, 255),
        'Green': (0, 255, 0),
        'Purple': (128, 0, 128),
        'Blue': (0, 0, 255),
        'Cyan': (0, 255, 255),
        'LightCyan': (224, 255, 255),
        'LightGreen': (144, 238, 144)
    }

    def __call__(self, red: int, green: int, blue: int) -> Tuple[float, float, float]:
        return red / 255, green / 255, blue / 255

    @classmethod
    def get(cls, name: str) -> Tuple[float, float, float]:
        return cls.__call__(*cls._color[name])


RGB = (palette := Palette())

edge = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe %s'
chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'


# : display
# ===============================================================================================================
def display(path: str = 'export/', name: str = 'figure'):
    plot.savefig(path + f'{name}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    # plot.savefig(path + 'figure.svg', dpi=600, bbox_inches='tight', pad_inches=0.1)
    # webbrowser.get(chrome).open(os.getcwd() + r'\export\figure.svg', new=1)

# Line: 157, 7, 63 | Face: 220, 183, 199
# Line: 192, 43, 78 | Face: 237, 193, 202
# Line: 218, 76, 75 |
