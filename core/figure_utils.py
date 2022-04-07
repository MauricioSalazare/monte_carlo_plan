import matplotlib
from distutils.version import LooseVersion

def set_figure_art(fontsize=7):
    # fontsize = 7
    linewidth = 0.4
    linewidth_lines = 0.8
    usetex=True
    matplotlib.rc('legend', fontsize=fontsize, handlelength=3)
    matplotlib.rc('axes', titlesize=fontsize)
    matplotlib.rc('axes', labelsize=fontsize)
    matplotlib.rc('axes', linewidth=linewidth)
    matplotlib.rc('patch', linewidth=linewidth)
    matplotlib.rc('hatch', linewidth=linewidth)
    matplotlib.rc('xtick', labelsize=fontsize)
    matplotlib.rc('xtick.major', width=0.4)
    matplotlib.rc('ytick.major', width=0.4)
    matplotlib.rc('ytick', labelsize=fontsize)
    matplotlib.rc('lines', linewidth=linewidth_lines)
    matplotlib.rc('text', usetex=usetex)
    matplotlib.rc('font', size=fontsize, family='serif',
                  style='normal', variant='normal',
                  stretch='normal', weight='normal')
    matplotlib.rc('patch', force_edgecolor=True)
    if LooseVersion(matplotlib.__version__) < LooseVersion("3.1"):
        matplotlib.rc('_internal', classic_mode=True)
    else:
        # New in mpl 3.1
        matplotlib.rc('scatter', edgecolors='b')
    matplotlib.rc('grid', linestyle=':', linewidth=linewidth)
    matplotlib.rc('errorbar', capsize=3)
    matplotlib.rc('image', cmap='viridis')
    matplotlib.rc('axes', xmargin=0)
    matplotlib.rc('axes', ymargin=0)
    matplotlib.rc('xtick', direction='in')
    matplotlib.rc('ytick', direction='in')
    matplotlib.rc('xtick', top=True)
    matplotlib.rc('ytick', right=True)
    # rcdefaults() # Reset the default settings of Matplotlib
    # plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])