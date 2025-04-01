"""
 +----------------+
 |   PLOTTER.PY   |
 +----------------+
Author: Salvador Blasco <salvador.blasco@gmail.com
"""


import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets as QtGui

from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import libsod


class Plotter(QtGui.QWidget):
    '''Widget to plot data.

    Pop-up non-modal widget where data is plotted. Data are fed through
    the feed method and plotted through the plot method.
    '''

    PLOT_NORMAL = 0
    PLOT_RECIPROCAL = 1
    PLOT_LOG = 2
    styles = 'o', '^', 'v', 'd', 's'
    colors = ('b', 'r'), ('g', 'r'), ('k', 'r')
    plotargs = {'ecolor': 'k', 'elinewidth': 1.0, 'capsize': 1.0,
                'barsabove': True}

    def __init__(self, parent=None):
        'Initialise class.'
        super().__init__()
        self.setWindowTitle("Plot")

        width = 5
        height = 4
        dpi = 100

        self._layout = QtGui.QVBoxLayout(self)
        self._figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self.canvas, self)

        self._layout.addWidget(self.canvas)
        self._layout.addWidget(self._toolbar)

        self.setLayout(self._layout)

        self._expdata = None
        self._fitdata = None
        self._ic50 = None
        self.__plottype = self.PLOT_NORMAL

        self._expdatapaths = []
        self._experrorpaths = []
        self._fitdatapaths = None

        self._build_popupmenu()

    def feed(self, experimental_data=None, fit_data=None, ic50=None):
        """Give data to work with.

        This method must be called before :func:`plot`

        Parameters:
            experimental_data (tuple): containing in turn (1) concentration,
                (2) inhibition percent, (3) error for inhibition concentration
                and (4) the use flag.
            fit_data (tuple): The x (concentration) and y (fitted inhibition
                concentration) values.
            ic50 (float): The value for the IC50
        """
        self._expdata = experimental_data
        self._fitdata = fit_data
        self._ic50 = ic50

    # def set_linear(self, linear=False):
    #     """Change plot type.

    #     Parameters:
    #         linear (bool): If True, a double reciprocal plot is drawn, else
    #             a regular plot is drawn.
    #     """
    #     # self.__linear = linear
    #     self.__plottype = self.PLOT_NORMAL
    #     self.plot()
    #     self.canvas.draw()

    def plot(self):
        """Clear canvas and plot data.

        Raises:
            RuntimeError: if no data has been fed previously.
        """
        if self._expdata is None:
            raise RuntimeError('No data has been provided for plotting.')

        self._figure.clear()
        axes = self._figure.gca()
        if self.__plottype == self.PLOT_LOG:
            axes.set_xscale('log')
        else:
            axes.set_xscale('linear')

        if self._expdata is not None:
            self.__plot_expdata()

        if self._fitdata is not None:
            # if self.__linear:
            #     x = np.reciprocal(self._fitdata[0][1:])
            #     y = np.reciprocal(self._fitdata[1][1:])
            # else:
            #     x = self._fitdata[0]
            #     y = self._fitdata[1]
            x, y = self.__xyfit()
            self._fitdatapaths = axes.plot(x, y, zorder=2)

        addtxt = '1 / ' if self.__plottype == self.PLOT_RECIPROCAL else ''
        axes.set_ylabel(addtxt + '% inhibition')
        axes.set_xlabel(addtxt + 'concentration')

        if self._ic50:
            self.__plot_ic50()

    def update_plot(self):
        "Update plot when only expdata change."
        if self._expdata is None:
            return

        sdata = self.__splitdata1(*self._expdata)
        for paths, dset in zip(self._expdatapaths, sdata):
            if self.__linear:
                paths.set_xdata = np.reciprocal(dset[0])
                paths.set_ydata = np.reciprocal(dset[1])
            else:
                paths.set_xdata = dset[0]
                paths.set_ydata = dset[1]
        self.canvas.draw()
        # self.draw()

    def _plot_ignored(self):
        self.plot()
        self.canvas.draw()

    def _popupmenu(self, point):
        self.popupmenu.popup(self.mapToGlobal(point))

    def _build_popupmenu(self):
        def _myaction(text, parent, checkable, checked, trig_action):
            action = QtGui.QAction(text, parent)
            action.setCheckable(checkable)
            action.setChecked(checked)
            action.triggered.connect(trig_action)
            return action

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._popupmenu)
        self.popupmenu = QtGui.QMenu(parent=self)

        self.action_plot_ignored = _myaction('Plot ignored', None, True, True,
                                             self._plot_ignored)
        self.popupmenu.addAction(self.action_plot_ignored)

        menu_plottype = QtGui.QMenu('plot type', self.popupmenu)

        action_plotnormal = _myaction('Normal', menu_plottype, True,
                                      self.__plottype == self.PLOT_NORMAL,
                                      self.__setnorm)
        action_plotlinear = _myaction('Linear', menu_plottype, True,
                                      self.__plottype == self.PLOT_RECIPROCAL,
                                      self.__setrecip)
        action_plotlog = _myaction('Logarithmic', menu_plottype, True,
                                   self.__plottype == self.PLOT_LOG,
                                   self.__setlog)

        group_plottype = QtGui.QActionGroup(menu_plottype)
        for action in (action_plotlinear, action_plotnormal, action_plotlog):
            group_plottype.addAction(action)
            menu_plottype.addAction(action)

        self.popupmenu.addMenu(menu_plottype)

    def __plot_expdata(self):
        "Plot experimental data."
        from itertools import cycle
        _ls = cycle(Plotter.styles)
        _co = cycle(Plotter.colors)
        axes = self._figure.gca()

        self._expdatapaths = []
        self._experrorpaths = []

        for c, ic, eic, u in libsod.splitdata1(*self._expdata):
            style = next(_ls)
            if self.action_plot_ignored.isChecked():
                dzip = zip(libsod.splitdata2(u, c),
                           libsod.splitdata2(u, ic),
                           libsod.splitdata2(u, eic),
                           next(_co))
            else:
                dzip = zip((libsod.splitdata2(u, c)[0],),
                           (libsod.splitdata2(u, ic)[0],),
                           (libsod.splitdata2(u, eic)[0],),
                           next(_co))
            for c_, ic_, eic_, color in dzip:
                if len(c_) == 0:
                    continue
                if self.__plottype == self.PLOT_RECIPROCAL:
                    path = axes.scatter(np.reciprocal(c_),
                                        np.reciprocal(ic_), zorder=5)
                else:
                    path = axes.errorbar(x=c_, y=ic_, yerr=eic_,
                                         linestyle='None', color=color,
                                         marker=style, **Plotter.plotargs)
                self._expdatapaths.append(path)

    def __plot_ic50(self):
        axes = self._figure.gca()
        # tr_a = axes.transAxes.transform
        # itr_a = axes.transAxes.inverted().transform
        # tr_d = axes.transData.transform
        # # itr_d = axes.transData.inverted().transform
        # _x50, _y50 = itr_a(tr_d((self._ic50, 50.0)))
        # print(_x50, _y50)
        # axes.axhline(y=50.0, xmin=0, xmax=_x50)
        # axes.axvline(x=self._ic50, ymin=0, ymax=_y50)

        x, y = self._ic50, 50.0
        if self.__plottype == self.PLOT_RECIPROCAL:
            x, y = 1/x, 1/y
        axes.scatter([x], [y], c='r', s=50, marker='+', linewidths=2.0,
                     zorder=10)

    def __replot(self):
        self.plot()
        self.canvas.draw()

    def __setrecip(self):
        "Make double reciprocal plot."
        self.__plottype = self.PLOT_RECIPROCAL
        self.__replot()

    def __setnorm(self):
        "Make normal plot."
        self.__plottype = self.PLOT_NORMAL
        self.__replot()

    def __setlog(self):
        "Make plot logarithmic."
        self.__plottype = self.PLOT_LOG
        self.__replot()

    def __xyfit(self):
        if self.__plottype == self.PLOT_RECIPROCAL:
            x = np.reciprocal(self._fitdata[0][1:])
            y = np.reciprocal(self._fitdata[1][1:])
        else:
            x = self._fitdata[0]
            y = self._fitdata[1]
        return x, y
