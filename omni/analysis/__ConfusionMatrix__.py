from ..plotting import *
from typing import TypeAlias, List, Optional
import numpy as np

array: TypeAlias = np.ndarray


class ConfusionMatrix(object):

    def __init__(self, outputs: array, targets: array, *,
                 category: Optional[List[str]] = None):
        assert outputs.shape == targets.shape
        _, num_of_classes = targets.shape
        cm = np.zeros((num_of_classes, num_of_classes))
        i = np.argmax(targets, axis=1)
        j = np.argmax(outputs, axis=1)
        pos = np.stack((i, j), axis=1)
        for i, j in pos:
            cm[i, j] += 1
        self._cm = cm.astype(int)

        self._num_of_classes = num_of_classes

        if category is None:
            category = [f'C{k}' for k in range(num_of_classes)]
        self._category = category

        self._size_of_figure = 8.0
        self._font_size_of_label = 8
        self._font_size_of_value = 6

    def show(self):
        data = self._cm.astype(float) / self._cm.sum(axis=1)[:, None] * 100

        plot.rcParams['xtick.direction'] = 'out'
        plot.rcParams['ytick.direction'] = 'out'

        figure: Figure = plot.figure(figsize=(8.0, 8.0))
        axes: Axes = figure.add_subplot()

        axes.set_xticks(list(range(self._num_of_classes)))
        axes.set_yticks(list(range(self._num_of_classes)))
        axes.set_xticklabels(self._category, fontdict={'fontsize': 8},
                             rotation=45,
                             rotation_mode='anchor',
                             ha='right')
        axes.set_yticklabels(self._category, fontdict={'fontsize': 8})
        axes.tick_params(width=0.9, colors=RGB(38, 38, 38))
        axes.set_xlabel('Predicted', fontdict={'fontsize': 11})
        axes.set_ylabel('True', fontdict={'fontsize': 11})
        axes.set_aspect('equal')

        image: AxesImage = axes.imshow(data, vmin=0, vmax=100,
                                       interpolation='nearest',
                                       cmap=plot.cm.get_cmap('Blues', 100))

        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        colorbar: Colorbar = plot.colorbar(image, cax=cax)
        colorbar.set_label('Accuracy (%)', fontsize=15)
        colorbar.set_ticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
        colorbar.set_ticklabels(['10', '20', '30', '40', '50', '60', '70', '80', '90'])
        colorbar.ax.tick_params(labelsize=10, colors=RGB(38, 38, 38))

        for k in range(self._num_of_classes):
            accuracy = data[k, k]
            axes.text(k, k + 1 / 30, '{:.2f}'.format(accuracy) if accuracy < 100.0 else '100.0',
                      color='white' if accuracy >= 35.0 else 'black',
                      fontsize=6,
                      ha='center',
                      va='center')

        plot.tight_layout()
        display(name='cm')
        plot.show()

    def get(self) -> array:
        return self._cm
