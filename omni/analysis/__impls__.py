from omni import *
from omni.plotting import *


# : plot_constellation
# ==========================================================================================================
def plot_constellation(modulator: comm.Modulator):
    symbols = np.asarray(list(modulator.mapping.values()))
    bound = np.max(np.abs(np.concatenate([symbols.real, symbols.imag]))) * 1.318
    figure: Figure = plot.figure(dpi=100)
    axes: Axes = figure.add_subplot()
    axes.set_xlim(-bound, bound)
    axes.set_ylim(-bound, bound)
    axes.set_xlabel('In-Phase')
    axes.set_ylabel('Quadrature')
    axes.set_title('Constellation Diagram')
    axes.set_aspect('equal')
    axes.tick_params(top=True, bottom=True, left=True, right=True)
    axes.tick_params(labeltop=False, labelbottom=True, labelleft=True, labelright=False)
    for symbol, value in modulator.mapping.items():
        axes.scatter(value.real, value.imag, s=18, color='r', edgecolor=RGB(38, 38, 38))
        axes.text(value.real + 0.15, value.imag - 0.25, str(symbol), color='k')
    axes.grid(True, which='major', linewidth=0.5, color=RGB(223, 223, 223))
    axes.set_axisbelow(True)
    plot.tight_layout()
    plot.show()


# : plot_waveform
# ==========================================================================================================
def plot_waveform(signal: array):
    assert signal.dtype == complex
    data = np.concatenate([signal.real, signal.imag])
    lbound, ubound = np.min(data) - 0.2, np.max(data) + 0.2

    figure, subplots = plot.subplots(2, 1, figsize=(8, 5), sharex='all', sharey='all')

    axes: Axes = subplots[0]
    axes.set_xlim(0, 257 if len(signal) >= 258 else len(signal) - 1)
    axes.set_ylim(lbound, ubound)
    axes.set_xticks([32, 64, 96, 128, 160, 192, 224])
    axes.set_xticklabels(['32', '64', '96', '128', '160', '192', '224'])
    axes.set_title('In-Phase')
    axes.plot(signal.real, color=RGB(97, 149, 200))
    axes.grid(True, which='major', linewidth=0.5, color=RGB(223, 223, 223))
    axes.set_axisbelow(True)

    axes: Axes = subplots[1]
    axes.set_title('Quadrature')
    axes.plot(signal.imag, color=RGB(255, 99, 71))
    axes.grid(True, which='major', linewidth=0.5, color=RGB(223, 223, 223))
    axes.set_axisbelow(True)
    plot.subplots_adjust(hspace=0.25)
    plot.tight_layout()
    plot.show()
