import matplotlib
from distutils.version import LooseVersion

def set_figure_art():
    """
    Reference for fontsize in matplotlib:
    https://stackoverflow.com/questions/62288898/matplotlib-values-for-the-xx-small-x-small-small-medium-large-x-large-xx

    xx-small    5.79
    x-small     6.94
    small       8.33
    medium      10.0
    large       12.0
    x-large     14.4
    xx-large    17.28
    larger      12.0
    smaller     8.33

    To reset everything in matplotlib use:

    from matplotlib import rcdefaults
    rcdefaults()
    """

    fontsize=6
    linewidth = 0.4
    grid_linewidth = 0.25
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
    matplotlib.rc('lines', linewidth=linewidth)
    matplotlib.rc('text', usetex=usetex)
    matplotlib.rc('font', size=fontsize, family='serif', serif=['Palatino'],
                  style='normal', variant='normal',
                  stretch='normal', weight='normal')
    # matplotlib.rc('font', size=fontsize, family='serif',
    #               style='normal', variant='normal',
    #               stretch='normal', weight='normal')  # This is the font for conferences ?
    # matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    # matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})  # This is the font of IEEE Journal
    matplotlib.rc('patch', force_edgecolor=True)
    if LooseVersion(matplotlib.__version__) < LooseVersion("3.1"):
        matplotlib.rc('_internal', classic_mode=True)
    else:
        # New in mpl 3.1
        matplotlib.rc('scatter', edgecolors='b')
    matplotlib.rc('grid', linestyle=':')
    matplotlib.rc('errorbar', capsize=3)
    matplotlib.rc('image', cmap='viridis')
    matplotlib.rc('axes', xmargin=0)

    matplotlib.rc('axes', grid=True)
    matplotlib.rc('axes', ymargin=0)
    matplotlib.rc('axes', ymargin=0)
    matplotlib.rc('grid', linewidth=grid_linewidth)
    matplotlib.rc('grid', linestyle='--')

    matplotlib.rc('xtick', direction='in')
    matplotlib.rc('ytick', direction='in')
    matplotlib.rc('xtick', top=True)
    matplotlib.rc('ytick', right=True)

    matplotlib.rc('lines', markersize=2)
    # rcdefaults() # Reset the default settings of Matplotlib
    # plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
