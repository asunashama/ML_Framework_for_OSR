from omni import *
from omni.plotting import *

_Rainbow_ = [
    RGB(205, 109, 103),
    RGB(204, 185, 103),
    RGB(148, 205, 103),
    RGB(103, 204, 135),
    RGB(103, 198, 205),
    RGB(103, 123, 205),
    RGB(160, 103, 204),
    RGB(204, 104, 173)
]


# : plot_loss
# ==========================================================================================================
def plot_loss(history: utils.History, *items: str, height: float = 7.0):
    figure: Figure = plot.figure(figsize=(5.2, 3.8))
    axes: Axes = figure.add_subplot()

    axes.set_xlabel('Epoch')
    axes.set_xlim(0, len(history.epoch) + 1)
    axes.set_ylim(0, height)
    axes.set_xticks([50, 100, 150, 200, 250, 300, 350])
    axes.set_xticklabels(['50', '100', '150', '200', '250', '300', '350'])
    axes.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
    axes.set_yticklabels(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0'])
    axes.tick_params(axis='both', width=0.5, colors=RGB(38, 38, 38))
    axes.set_axisbelow(True)
    axes.grid(True, which='major', linewidth=0.5, color=RGB(223, 223, 223))

    for k, name in enumerate(items):
        axes.plot(history.epoch, history[name], label=name, color=_Rainbow_[-k])

    axes.legend(loc='upper right',
                bbox_to_anchor=(0.98, 0.98),
                fancybox=False,
                edgecolor=RGB(38, 38, 38),
                handlelength=3.5,
                fontsize=10)

    plot.tight_layout()
    plot.show()


# : plot_accuracy
# ==========================================================================================================
def plot_accuracy(history: utils.History, *items: str):
    figure: Figure = plot.figure(figsize=(5.2, 3.8))
    axes: Axes = figure.add_subplot()

    axes.set_xlabel('Epoch')
    axes.set_ylabel('Accuracy (%)')
    axes.set_xlim(0, len(history.epoch) + 1)
    axes.set_ylim(7, 103)
    axes.set_xticks([10, 30, 50, 70, 90, 110, 130, 150, 170])
    axes.set_xticklabels(['10', '30', '50', '70', '90', '110', '130', '150', '170'])
    axes.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    axes.set_yticklabels(['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
    axes.tick_params(width=0.5, colors=RGB(38, 38, 38))
    axes.set_axisbelow(True)
    axes.grid(True, linewidth=0.5, color=RGB(223, 223, 223))

    for k, name in enumerate(items):
        axes.plot(history.epoch, np.asarray(history[name]) * 100, label=name, color=_Rainbow_[k])

    axes.legend(loc='lower right',
                bbox_to_anchor=(0.98, 0.04),
                fancybox=False,
                edgecolor=RGB(38, 38, 38),
                handlelength=3.5,
                fontsize=10)

    plot.tight_layout()
    plot.show()


# ... ConfusionMatrix
# ==========================================================================================================
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

    def show(self, *, export: str = 'export/cm.pdf'):
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
        if export is not None:
            plot.savefig(export, dpi=600, bbox_inches='tight', pad_inches=0.05)
        plot.show()

    @property
    def data(self) -> array:
        return self._cm


# ... Embedding
# ==========================================================================================================
class Embedding(object):
    _ALLOCATOR = {
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

    def __init__(self, points: array, labels: array, allocator: Optional[Dict[str, RGB]] = None):
        self._points = points
        self._labels = labels.flatten()
        self._allocator = self._ALLOCATOR if allocator is None else allocator

    def show(self, observe: Optional[int] = None, *, export: str = 'export/embedding.pdf'):
        figure: Figure = plot.figure()
        axes: Axes = figure.add_subplot()

        axes.set_xticks([])
        axes.set_yticks([])
        axes.tick_params(axis='both', width=0.9, colors=RGB(38, 38, 38))
        axes.set_axisbelow(True)
        axes.grid(True, which='major', linewidth=0.5, color=RGB(223, 223, 223))

        for k, name in enumerate(self._allocator):
            if k == observe:
                axes.scatter(self._points[self._labels == k, 0],
                             self._points[self._labels == k, 1],
                             color='none', label='target', marker='o', edgecolor='k', s=10)
            else:
                axes.scatter(self._points[self._labels == k, 0],
                             self._points[self._labels == k, 1],
                             color=self._allocator[name], label=name, s=8)

        axes.axis('tight')
        axes.legend()

        plot.get_current_fig_manager().set_window_title('Embedding Space')

        plot.tight_layout()
        if export is not None:
            plot.savefig(export, dpi=600, bbox_inches='tight', pad_inches=0.05)
        plot.show()
