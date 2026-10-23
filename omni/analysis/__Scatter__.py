from omni import *
from omni.plotting import *

allocator = {
    'C01': RGB(239, 52, 52),
    'C02': RGB(255, 135, 0),
    'C03': RGB(255, 211, 0),
    'C04': RGB(222, 255, 10),
    'C05': RGB(161, 255, 10),
    'C06': RGB(10, 255, 153),
    'C07': RGB(20, 125, 245),
    'C08': RGB(26, 148, 188),
    'C09': RGB(88, 10, 255),
    'C10': RGB(88, 160, 75),
    'C11': RGB(90, 110, 255),
    'C12': RGB(30, 210, 255),
    'C13': RGB(190, 70, 155),
    'C14': RGB(188, 60, 75),
    'C15': RGB(108, 160, 175),
    'C16': RGB(78, 221, 35),
    'C17': RGB(58, 60, 175),
    'C18': RGB(201, 160, 85)
}


class Space(object):

    def __init__(self, points: array, labels: array, allocator: Dict):
        self._points = points
        self._labels = labels.flatten()
        self._allocator = allocator

    def show(self, observe: int, path='export/'):
        figure: Figure = plot.figure()
        axes: Axes = figure.add_subplot()

        axes.set_xticks([])
        axes.set_yticks([])
        axes.tick_params(axis='both', width=0.9, colors=RGB(38, 38, 38))
        axes.set_axisbelow(True)
        axes.grid(True, which='major', linewidth=0.5, color=RGB(223, 223, 223))

        for k, name in enumerate(allocator):
            axes.scatter(self._points[self._labels == k, 0],
                         self._points[self._labels == k, 1],
                         color=allocator[name], label=name)

        # axes.scatter(embedded_features[y == 13, 0],
        #              embedded_features[y == 13, 1],
        #              color='none', label='target', marker='o', edgecolor='k')

        axes.axis('tight')
        axes.legend()
        plot.tight_layout()
        display(path='export/', name='')
        plot.show()
