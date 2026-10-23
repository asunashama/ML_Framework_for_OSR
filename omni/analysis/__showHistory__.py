from omni import *
from omni.plotting import *


# : plot_loss
# ===============================================================================================================
def plot_loss(history: utils.History):
    figure: Figure = plot.figure(figsize=(5.2, 3.8))
    axes: Axes = figure.add_subplot()

    axes.set_xlabel('Epoch')
    axes.set_xlim(0, len(history.epoch) + 1)
    axes.set_ylim(0, 9)
    axes.set_xticks([10, 30, 50, 70, 90, 110, 130, 150, 170])
    axes.set_xticklabels(['10', '30', '50', '70', '90', '110', '130', '150', '170'])
    axes.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
    axes.set_yticklabels(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0'])
    axes.tick_params(axis='both', width=0.5, colors=RGB(38, 38, 38))
    axes.set_axisbelow(True)
    axes.grid(True, which='major', linewidth=0.5, color=RGB(223, 223, 223))
    axes.plot(history.epoch, history['loss'],
              label='Training Loss', color=RGB(239, 52, 52))
    axes.plot(history.epoch, history['val_loss'],
              label='Validation Loss', color=RGB(75, 99, 170))
    axes.legend(loc='upper right',
                bbox_to_anchor=(0.98, 0.98),
                fancybox=False,
                edgecolor=RGB(38, 38, 38),
                handlelength=3.5,
                fontsize=10)
    plot.tight_layout()
    plot.show()


# : plot_accuracy
# ===============================================================================================================
def plot_accuracy(history: utils.History):
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
    axes.plot(history.epoch, np.asarray(history['accuracy']) * 100,
              label='accuracy', color=RGB(239, 52, 52))
    axes.plot(history.epoch, np.asarray(history['val_accuracy']) * 100,
              label='val_accuracy', color=RGB(52, 52, 239))
    axes.legend(loc='lower right',
                bbox_to_anchor=(0.98, 0.04),
                fancybox=False,
                edgecolor=RGB(38, 38, 38),
                handlelength=3.5,
                fontsize=10)
    plot.tight_layout()
    plot.show()
