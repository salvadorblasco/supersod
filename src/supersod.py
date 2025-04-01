#!/usr/bin/python3
'''
SuperSOD, an application for the treatment of SOD data.
Author: Salvador Blasco
Email:  salvador.blasco@gmail.com
(c) 2007-2025
'''

# pylint: disable=invalid-name

import sys

import numpy as np
from PyQt5 import Qt, QtCore
from PyQt5 import QtWidgets as QWidgets

import consts
import libio
import libqt
import libsod
import plotter
import ui_mainwindow as mainui
import ui_points as pointui
import ui_import as importui


__version__ = '1.1.1'

CELL_NONEDITABLE = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
CELL_EDITABLE = CELL_NONEDITABLE | QtCore.Qt.ItemIsEditable


class MainWindow(QWidgets.QMainWindow):
    "The Main Window."

    def __init__(self):
        'Initialise object.'
        super().__init__()

        self.ui = mainui.Ui_MainWindow()
        self.ui.setupUi(self)

        self.plotter = plotter.Plotter()
        self.data = {}
        self.indicator = consts.NBT
        self.cindicator = 50.0
        self.fit_params = {'f': 0.01, 'K': 1.0, 'error_f': 0.0, 'error_K': 0.0}
        self.default_dir = '.'
        self.current_file = None

        self.columnfields = ('conc', 'slope', 'error', 'IC', 'errIC',
                             'residual', 'weight')
        self.column = {k: v for v, k in enumerate(self.columnfields)}

        from PyQt5.QtGui import QBrush, QColor
        self._blankbrush = QBrush(QColor(Qt.Qt.lightGray))
        self.setWindowTitle('SuperSOD, v%s' % __version__)

        self.__build_popupmenu()
        self.__connections()
        self._readconfig()

    def addPoint(self):
        '''Add a new row to the main table.

        Add a new point below currently selected row or at the bottom
        if no row is selected.
        '''
        current_row = self.ui.table.currentRow()
        self.ui.table.insertRow(current_row+1)
        for col in range(self.ui.table.columnCount()):
            widget = QWidgets.QTableWidgetItem()
            self.ui.table.setItem(current_row+1, col, widget)

    def append_data(self, filename=None):
        '''Open a .sod file and append its contents to current project.

        Parameters:
            filename (str): The name of the file to be openes. If None,
                dialog will pop up to retrieve a file name.
        '''
        if filename is None:
            filters = "SuperSOD Files (*.sod);;All Files (*.*)"
            filename = self.__opendialog(filters)
            if not filename:
                return

        self.current_file = filename
        stream = libio.open_sodfile(filename)
        new_indicator = consts.valid_indicators.index(next(stream))
        if new_indicator != self.indicator:
            libqt.popwarning('Import error',
                             'Loaded dataset uses a different indicator.')

        new_cindicator = float(next(stream))
        if new_cindicator != self.cindicator:
            msg = 'Loaded dataset uses a different indicator concentration.'
            libqt.popwarning('Import error', msg)

        info = tuple(stream)
        conc, means, stds, weight, use, rpoints = info
        self.data['raw_points'].extend(rpoints)
        self._lay_table(conc, means, stds, weight, use, extend=True)

    def cellChanged(self, row, col):
        '''Slot for when a cell is edited.

        Parameters:
            row (int): The row that has been edited.
            col (int): The column that has been edited.
        '''
        self.ui.table.cellChanged.disconnect()

        try:
            float(self.ui.table.item(row, col).text())
        except ValueError:
            self.ui.table.item(row, col).setText('????')
            self.ui.table.cellChanged.connect(self.cellChanged)
            return

        concentration = self._readnd('conc')
        if col == self.column['conc']:
            self.__stamp('conc', concentration, fmt='%.4f', editable=True)
        elif col in (self.column['slope'], self.column['error']):
            slope = self._readnd('slope')
            error = self._readnd('error')
            inhibitionc, err_inhibitionc = zip(*libsod.calc_IC(concentration,
                                                               slope, error))
            self.__stamp('IC', inhibitionc, fmt='%.2f', editable=False)
            self.__stamp('errIC', err_inhibitionc, fmt='%.2f', editable=False)

        self._color_row()
        self.ui.table.cellChanged.connect(self.cellChanged)

    def deletePoint(self):
        'Delete current row in the data table.'
        self.ui.table.cellChanged.disconnect()

        for row in reversed(sorted(libqt.selected_rows(self.ui.table))):
            self.ui.table.removeRow(row)

        self._titlerows()
        self._color_row()
        self.ui.table.cellChanged.connect(self.cellChanged)

    def editPoint(self):
        'Open dialog to edit current point.'
        if 'raw_points' not in self.data:
            self.data['raw_points'] = self.ui.table.rowCount() * [None]
        crow = self.ui.table.currentRow()
        if self.data['raw_points'][crow] is None:
            inhibitionc = tuple(self._readnd('IC'))
            self.data['raw_points'][crow] = [(True, None, inhibitionc[crow])]
        pointdialog = Points(data=self.data['raw_points'][crow], parent=self)
        ret = pointdialog.exec()
        if ret == QWidgets.QDialog.Accepted:
            self.data['raw_points'][crow] = pointdialog.data
            for value, tag in zip(pointdialog.average(), ('slope', 'error')):
                widget = self.ui.table.item(crow, self.column[tag])
                widget.setText('%.4f' % value)

    def exportData(self):
        header = "\t".join(( '"' + self.ui.table.horizontalHeaderItem(col).text() + '"'
                      for col in range(self.ui.table.columnCount())))
        output = "\n".join((
            "\t".join(self.ui.table.item(row, col).text() 
                      for col in range(self.ui.table.columnCount()))
            for row in range(self.ui.table.rowCount())))

        filters = "Text Files (*.txt);;All Files (*.*)"
        filename = self.__savedialog(filters)
        if filename:
            with open(filename, 'w', encoding="utf-8") as f:
                f.write(header + '\n' + output)

    def fit(self):
        """Fit data and update plot.

        This routine performs in turn: (1) collect data from main table,
        (2) fit the data and (3) update plot.
        """
        # collect data from table
        concentration = self._readnd('conc')
        inhibitionc = self._readnd('IC')
        err_inhibitionc = self._readnd('errIC')
        weight = self._readnd('weight')
        use_flag = np.logical_not(np.fromiter(self.__use(), dtype=bool))

        masked_concentration = np.extract(use_flag, concentration)
        masked_inhibitionc = np.extract(use_flag, inhibitionc)
        masked_weight = np.extract(use_flag, weight)

        # feed fitting function
        fit_output = libsod.fit(masked_concentration,
                                masked_inhibitionc/100.0,
                                masked_weight)
        ic50, error_ic50, par_f, error_f, par_K, error_K = fit_output

        # update table
        k, erk = libsod.calc_kcat(ic50, error_ic50,
                                  consts.k_indicator[self.indicator],
                                  consts.errk_indicator[self.indicator],
                                  self.cindicator, 0.0)
        msg = 'fit result: IC₅₀={:.4f}±{:.4f}µM, k={:.4e}±{:.4e}(Ms)^-1'
        _ = msg.format(ic50, error_ic50, k, erk)
        self.ui.statusbar.showMessage(_)

        keys = 'f', 'error_f', 'K', 'error_K', 'IC50', 'error_IC50', 'kcat', \
               'error_kcat'
        values = par_f, error_f, par_K, error_K, ic50, error_ic50, k, erk
        self.fit_params.update(zip(keys, values))
        fit_pars = par_f, par_K

        calcd_inhibitionc = libsod.f_IC(fit_pars, concentration)
        residuals = (a - 100*b for a, b in zip(inhibitionc, calcd_inhibitionc))

        self.ui.table.cellChanged.disconnect()
        libqt.fill_column(self.ui.table, self.column['residual'],
                          data=('%.2f' % s for s in residuals),
                          flags=CELL_NONEDITABLE)
        self._color_row()

        rows = libqt.iter_column(self.ui.table, col=0)
        for row in libqt.indices_crossed(rows):
            libqt.cross(libqt.iter_row(self.ui.table, row), to_cross=True)

        self.ui.table.cellChanged.connect(self.cellChanged)

        import math
        cexcluded = concentration[concentration>0.0]
        _min = math.log10(0.95*np.min(cexcluded))
        _max = math.log10(1.05*np.max(cexcluded))
        xfit = np.logspace(_min, _max, 50)
        yfit = 100*libsod.f_IC(fit_pars, xfit)
        self.plotter.feed(experimental_data=(concentration, inhibitionc,
                                             err_inhibitionc, use_flag),
                          fit_data=(xfit, yfit),
                          ic50=ic50)
        self.plotter.plot()
        if self.plotter.isVisible():
            self.plotter.canvas.draw()
        else:
            self.plotter.show()

    def ignorePoint(self):
        'Ignore or unignore currently selected row(s).'
        rows0 = libqt.selected_rows(self.ui.table)
        extra_rows = set()
        for row in rows0:
            if self._isblank(row):
                for crow in self._dataset_rows(row+1):
                    extra_rows.add(crow)
        rows = rows0 | extra_rows

        self.ui.table.cellChanged.disconnect()
        for row in rows:
            libqt.cross(libqt.iter_row(self.ui.table, row))
        self.ui.table.cellChanged.connect(self.cellChanged)

    def importData(self, filename=None):
        '''Import raw data from several files.

        Parameters:
            filename (str): The file to read data from. If None, a
                dialog will pop up to retrieve a file name.
        '''
        if filename is None:
            filters = "Agilent CSV files (*.*)", "Cary CSV files (*.*)"
            filename, chsfilt = libqt.opendialog(parent=self, filters=";;".join(filters),
                                                 directory=self.default_dir)
            if not filename:
                return
            csvformat: int = filters.index(chsfilt)

        try:
            datastream = libio.import_data(filename, csvformat)

            self.indicator = consts.valid_indicators.index(next(datastream))
            self.cindicator = next(datastream)

            conc, slope, err_slope, raw_points = \
                libsod.autoprocess_spectra(datastream)
        except FileNotFoundError as e:
            libqt.popwarning(
                "File not found",
                str(e))
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            libqt.popwarning(
                "Error while importing data",
                str(e))
        else:
            self.data['raw_points'] = raw_points
            self._lay_table(conc, slope, err_slope, None, None)

    def newProject(self):
        'Remove data from self and clear table.'
        table = self.ui.table
        table.cellChanged.disconnect()
        table.clear()
        self._tabheader()
        table.setRowCount(1)
        for col in range(table.columnCount()):
            table.setItem(0, col, QWidgets.QTableWidgetItem())
        table.cellChanged.connect(self.cellChanged)

        self.data = {}
        self.indicator = consts.NBT
        self.cindicator = 50.0
        self.current_file = None

    def openProject(self, filename=None):
        '''Open a .sod file and load its contents.

        Parameters:
            filename (str): The name of the file to be openes. If None,
                dialog will pop up to retrieve a file name.
        '''
        if filename is None:
            filters = "SuperSOD Files (*.sod);;All Files (*.*)"
            filename = self.__opendialog(filters)
            if not filename:
                return

        self.current_file = filename
        self.__save_current_dir(filename)
        stream = libio.open_sodfile(filename)
        self.indicator = consts.valid_indicators.index(next(stream))
        self.cindicator = float(next(stream))
        info = tuple(stream)
        conc, means, stds, weight, use, rpoints = info
        self.data['raw_points'] = rpoints
        self._lay_table(conc, means, stds, weight, use, extend=False)

    def saveProject(self, filename=None):
        '''Save current project to a file.

        Parameters:
            filename (str): If None, the data will be saved to
                :var:`self.current_file`. If this variable is None, then a
                dialog will pop up to retrieve a file name.
        '''
        if filename is None:
            if self.current_file is None:
                filters = "SuperSOD Files (*.sod);;All Files (*.*)"
                filename = self.__savedialog(filters)
                if not filename:
                    return
                else:
                    self.current_file = filename
            else:
                filename = self.current_file

        xmlstream = libio.save_sodfile(
            consts.valid_indicators[self.indicator],
            self.cindicator,
            tuple(self._read('conc')),
            tuple(self._read('slope', scale=1e-4)),
            tuple(self._read('error', scale=1e-4)),
            tuple(self._read('IC', scale=1e-2)),
            tuple(self._read('errIC', scale=1e-2)),
            tuple(not u for u in self.__use()),
            tuple(self._read('weight')),
            self.data['raw_points'])

        with open(filename, 'w') as fhandler:
            fhandler.write(xmlstream)

        self.__save_current_dir(filename)

    def saveProjectAs(self):
        '''Open dialog to save file.'''
        filters = "SuperSOD Files (*.sod);;All Files (*.*)"
        filename = self.__savedialog(filters)
        if not filename:
            return
        self.current_file = filename
        self.saveProject(filename=filename)

    def _about(self):
        'Pop up about information.'
        m = QWidgets.QMessageBox(parent=self)
        m.setIcon(QWidgets.QMessageBox.Information)
        m.setText('Supersod ' + __version__)
        info = '''\
        (c) 2007-2019
        by Salvador Blasco
        <salvador.blasco@gmail.com>'''
        m.setInformativeText(info)
        m.addButton(QWidgets.QMessageBox.Ok)
        m.exec_()

    def _close(self):
        self._writeconfig()
        self.plotter.close()
        self.close()

    def _color_row(self):
        "Paint row in gray if it is a blank."
        for row in range(self.ui.table.rowCount()):
            if self._isblank(row):
                for col in libqt.iter_row(self.ui.table, row):
                    col.setBackground(self._blankbrush)

    def _dataset_rows(self, row0):
        for row in range(row0, self.ui.table.rowCount()):
            if self._isblank(row):
                break
            else:
                yield row

    def _isblank(self, row):
        return float(self.ui.table.item(row, 0).text()) == 0.0

    def _lay_table(self, conc, slope, error_slope, weight=None, use=None,
                   extend=False):
        'Put data into table.'
        self.ui.table.cellChanged.disconnect()
        if extend:
            row0 = self.ui.table.rowCount()
            total_rows = row0 + len(conc)
        else:
            row0 = 0
            total_rows = len(conc)
            self.ui.table.clear()
        self.ui.table.setRowCount(total_rows)
        self._tabheader()

        ic, eic = zip(*libsod.calc_IC(conc, slope, error_slope))
        if weight is None:
            weight = libsod.quadratic_weighting_scheme(ic)
        if use is None:
            use = len(conc)*(True,)

        self.__stamp('conc', conc, row0=row0)
        self.__stamp('slope', slope, scale=10000, row0=row0)
        self.__stamp('error', error_slope, scale=10000, row0=row0)
        self.__stamp('IC', ic, fmt='%.2f', editable=False, row0=row0)
        self.__stamp('errIC', eic, fmt='%.2f', editable=False, row0=row0)
        self.__stamp('weight', weight, editable=True, row0=row0)
        self.__stamp('residual', total_rows*[0.0], editable=False, row0=row0)

        for row, usef in enumerate(use, start=row0):
            libqt.cross(libqt.iter_row(self.ui.table, row), to_cross=not usef)

        self._color_row()
        self._titlerows()
        self.ui.table.cellChanged.connect(self.cellChanged)

    def _report(self):
        "Pop up a window with the report."
        dialog = QWidgets.QDialog()
        layout = QWidgets.QVBoxLayout()
        textwidget = QWidgets.QTextEdit()
        layout.addWidget(textwidget)
        dialog.setLayout(layout)

        text = libsod.html_report(
            indicator=self.indicator,
            cindicator=self.cindicator,
            concentration=self._readnd('conc'),
            inhibition=self._readnd('IC'),
            slope=self._readnd('slope'),
            use=tuple(self.__use()),
            **self.fit_params
        )
        textwidget.setHtml(text)
        dialog.exec()

    def _titlerows(self):
        if self.ui.table.rowCount() == 1:
            return
        from string import ascii_uppercase
        # ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        blanks = [self._isblank(r) for r in range(self.ui.table.rowCount())]
        _ind = [i[0] for i in enumerate(blanks) if i[1]] + [len(blanks)]
        assert _ind[0] == 0
        ind2 = [i-j for i, j in zip(_ind[1:], _ind[:-1])]
        ret = []
        for letter, number in zip(ascii_uppercase, ind2):
            ret.extend([letter + str(i) for i in range(number)])
        self.ui.table.setVerticalHeaderLabels(ret)

    def _popupmenu(self, point):
        self.popupmenu.popup(self.ui.table.viewport().mapToGlobal(point))

    def _read(self, what, scale=None):
        contents = libqt.iter_column_text(self.ui.table, self.column[what])
        if scale is None:
            retv = contents
        else:
            assert isinstance(scale, float)
            retv = (float(s)*scale for s in contents)
        return retv

    def _readnd(self, what, dtype=float):
        return np.fromiter(self._read(what), dtype=dtype)

    def _readconfig(self):
        # global default_dir
        config_file = 'config.ini'
        import os.path
        if not os.path.exists(config_file):
            return

        import configparser
        config = configparser.ConfigParser()
        config.read(config_file)
        self.default_dir = config['PATH']['default_dir']

    def _writeconfig(self):
        # global default_dir
        config_file = 'config.ini'
        section_name = 'PATH'
        import configparser
        config = configparser.ConfigParser()
        if not config.has_section(section_name):
            config.add_section(section_name)
        config.set(section_name, 'default_dir', self.default_dir)
        with open(config_file, 'w') as f:
            config.write(f)

    def __build_popupmenu(self):
        self.ui.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.table.customContextMenuRequested.connect(self._popupmenu)
        self.popupmenu = QWidgets.QMenu(parent=self)
        labels = ('(Un)ignore', 'Delete', 'Add', 'Edit')
        slots = (self.ignorePoint, self.deletePoint, self.addPoint,
                 self.editPoint)
        for label, slot in zip(labels, slots):
            action = self.popupmenu.addAction(label)
            action.triggered.connect(slot)

    def __connections(self):
        self.__plottype = QWidgets.QActionGroup(self)
        self.__plottype.addAction(self.ui.actionPlotNormal)
        self.__plottype.addAction(self.ui.actionPlotLinearized)

        u = self.ui
        actions_slots = (
            (u.actionNewProject, self.newProject),
            (u.actionExit, self._close),
            (u.actionImportData, lambda: self.importData()),
            (u.actionAppendData, lambda: self.append_data()),
            (u.actionOpenProject, lambda: self.openProject()),
            (u.actionSaveProject, lambda: self.saveProject()),
            (u.actionSaveProjectAs, self.saveProjectAs),
            (u.actionExportData, self.exportData),
            (u.actionFitReport, self._report),
            (u.actionWeightUnit, self.__weight_unit),
            (u.actionWeightQuadratic, self.__weight_quadr),
            (u.actionAbout, self._about),
            (self.__plottype, self.__plotTypeChanged)
        )

        for action, slot in actions_slots:
            action.triggered.connect(slot)

        u.table.cellChanged.connect(self.cellChanged)
        u.btn_fit.clicked.connect(self.fit)

    def _tabheader(self):
        hlabels = ('C / µM', 'slope ·10⁴s', 'error · 10⁴s', 'IC / %',
                   'error IC / %', 'residual', 'weight')
        self.ui.table.setHorizontalHeaderLabels(hlabels)

    def __opendialog(self, filters):
        # global default_dir
        filename, _ = libqt.opendialog(parent=self, filters=filters,
                                       directory=self.default_dir)
        return filename

    def __plotTypeChanged(self, action):
        self.plotter.set_linear(action is self.ui.actionPlotLinearized)

    def __isPlotLinear(self):
        return self.__plottype.checkedAction() is self.ui.actionPlotLinearized

    def __savedialog(self, filters):
        filename, _ = QWidgets.QFileDialog.getSaveFileName(
            parent=self,
            caption='Choose file to save into',
            directory=self.default_dir,
            filter=filters)
        return filename

    def __save_current_dir(self, filename):
        import os.path
        # global default_dir
        self.default_dir = os.path.dirname(filename)

    def __stamp(self, lbl, item, fmt='%.4f', scale=1.0, editable=True, row0=0):
        _item = (fmt % (i*scale) for i in item)
        flags = CELL_EDITABLE if editable else CELL_NONEDITABLE
        libqt.fill_column(self.ui.table, col=self.column[lbl],
                          data=_item, flags=flags, row0=row0)

    def __use(self):
        'Return use flags.'
        items = libqt.iter_column(self.ui.table, col=0)
        return libqt.bool_crossed(items)

    def __weight_aux(self, weight):
        libqt.fill_column(self.ui.table, col=self.column['weight'],
                          data=('%.4f' % w for w in weight))
        self._color_row()

    def __weight_unit(self):
        unit_weight = self.ui.table.rowCount() * (1.0,)
        self.__weight_aux(unit_weight)

    def __weight_quadr(self):
        ic = (float(s) for s in self._read('IC'))
        weight = libsod.quadratic_weighting_scheme(ic)
        self.__weight_aux(weight)


class Points(QWidgets.QDialog):
    'Dialog for the edition of individual measures.'

    def __init__(self, data, parent=None):
        '''Initialize widget.

        Parameters:
            data: A :class:`list` containing the following :
                - use (bool)
                - (lower_cutoff, upper_cutoff): the lower and upper bounds for
                    the linear fit.
                - rawdata: the arrays containing the raw spectrum or the
                    value of the slope.
        '''
        super().__init__(parent=parent)
        self.default_dir = parent.default_dir if parent is not None else '.'

        self.ui = pointui.Ui_PointDialog()
        self.ui.setupUi(self)
        self.setWindowTitle('Refine points')

        self.data = data
        self.__cutoff = [d[1] for d in data]
        self.__rawdata = [d[2] for d in data]

        self._filltable(data)
        self.__build_popupmenu()

        self.ui.pb_accept.clicked.connect(self.accept)
        self.ui.pb_cancel.clicked.connect(super().reject)

        self._set_average()

        self.ui.table.cellChanged.connect(self.cellChanged)

    def accept(self):
        'Accept for, update data and close.'
        new_data = []
        crossed = tuple(libqt.indices_crossed(libqt.iter_column(self.ui.table,
                                                                col=0)))
        use = (i in crossed for i in range(self.ui.table.rowCount()))
        ic = (float(i) for i in libqt.iter_column_text(self.ui.table, col=0))
        for u, c, r, i in zip(use, self.__cutoff, self.__rawdata, ic):
            if c is None:
                new_data.append((not u, None, i))
            else:
                new_data.append((not u, c, r))
        self.data = new_data
        super().accept()

    def average(self):
        '''Average and standard deviation of data.

        Returns:
            tuple: the average and the standard deviation
        '''
        import statistics
        d = tuple(self.slopes())
        if len(d) > 1:
            avg, std = statistics.mean(d), statistics.stdev(d)
        if len(d) == 1:
            avg, std = d[0], 0.0
        if not d:
            avg, std = 0.0, 0.0
        return avg, std

    def slopes(self, filter_crossed=True):
        '''Read slope values from table and return.

        Parameters:
            filter_crossed (bool): If True, do not include those values that
                are crossed. If False, all values are returned.
        Yields:
            The slope values.
        '''
        if filter_crossed:
            items = libqt.filter_crossed(
                libqt.iter_column(self.ui.table, col=0), invert=True)
        else:
            items = libqt.iter_column(self.ui.table, col=0)
        yield from (float(i.text()) for i in items)

    def cellChanged(self, row, col):
        '''Slot for when a cell is edited.

        Parameters:
            row (int): The row that has been edited.
            col (int): The column that has been edited.
        Raises:
            RuntimeError if a non-editable cell is passed.
        '''
        self.ui.table.cellChanged.disconnect()
        if col == 1 or col == 2:
            raise RuntimeError
        if self.__cutoff[row] is not None:
            self.__cutoff[row] = None
            self.__rawdata[row] = None
            widget = self.ui.table.item(row, 1)
            widget.setText('-')

        self._set_average()
        self._do_q()
        self.ui.table.cellChanged.connect(self.cellChanged)

    def _filltable(self, data):
        def _process_rawdata(use, lims, d):
            import scipy.stats as spst
            if lims is None:
                _slopes = d
                _r2 = '-'
            else:
                x = d[0][lims[0]:lims[1]]
                y = d[1][lims[0]:lims[1]]
                slope, _, r, _, _ = spst.linregress(x, y)
                _slopes = 1e4 * slope
                _r2 = '%.5f' % (r**2)
            return use, _slopes, _r2

        _use, slopes, r2 = zip(*[_process_rawdata(*q) for q in data])

        self.ui.table.setRowCount(len(data))
        assert len(slopes) == len(r2) == len(_use) == self.ui.table.rowCount()

        libqt.fill_column(self.ui.table, col=0,
                          data=('%.5f' % s for s in slopes),
                          flags=CELL_EDITABLE)
        libqt.fill_column(self.ui.table, col=1, data=r2,
                          flags=CELL_NONEDITABLE)

        items = (self.ui.table.item(row, col)
                 for row, u in enumerate(_use)
                 if not u
                 for col in (0, 1))
        libqt.cross(items)

        self._do_q()
        self.__tableheaderq()

    def _do_q(self):
        import libstat
        n = self.ui.table.rowCount()
        qtext = n*['-', ]
        slopes = tuple(self.slopes(filter_crossed=True))
        if len(slopes) > 2:
            full_slopes = tuple(self.slopes(filter_crossed=False))
            q = libstat.dixons_Q(slopes)
            qtext[full_slopes.index(min(slopes))] = '%.3f' % q[0]
            qtext[full_slopes.index(max(slopes))] = '%.3f' % q[1]

        libqt.fill_column(self.ui.table, col=2, data=qtext,
                          flags=CELL_NONEDITABLE)

    def _popupmenu(self, point):
        self.popupmenu.popup(self.ui.table.viewport().mapToGlobal(point))

    def _set_average(self):
        txt = '{:.4f} ± {:.4f}'
        self.ui.lbl_mean.setText(txt.format(*self.average()))

    def __build_popupmenu(self):
        self.ui.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.table.customContextMenuRequested.connect(self._popupmenu)
        self.popupmenu = QWidgets.QMenu(parent=self)
        labels = ('(Un)ignore', 'Delete', 'New Point', 'New from file',
                  'Edit', 'Autofit')
        slots = (self.ignorePoint, self.deletePoint, self.addPoint,
                 self.add_fromfile, self.editPoint, self.autoFit)
        for label, slot in zip(labels, slots):
            action = self.popupmenu.addAction(label)
            action.triggered.connect(slot)

    def __tableheaderq(self):
        import libstat
        n = libqt.count_crossed(libqt.iter_row(self.ui.table, row=0),
                                invert=True)
        if n > 2:
            q = libstat.dixon_critical(number_samples=n, confidence=95)
            widget = QWidgets.QTableWidgetItem('Q = %.3f' % q)
        else:
            widget = QWidgets.QTableWidgetItem('Q')
        self.ui.table.setHorizontalHeaderItem(2, widget)

    def autoFit(self):
        'Feed data to :func:`libsod.auto_fit` routine and update.'
        import scipy.stats as spst
        table = self.ui.table
        table.cellChanged.disconnect()
        for row in libqt.selected_rows(table):
            x, y = self.__rawdata[row][0], self.__rawdata[row][1]
            lower, upper, _ = libsod.auto_fit(x, y)
            slope, _, r_coef, _, _ = spst.linregress(x[lower:upper],
                                                     y[lower:upper])
            self.__cutoff[row] = (lower, upper)
            widget_slope = table.item(row, 0)
            widget_slope.setText('%.5f' % (1e4 * slope))
            widget_r = table.item(row, 1)
            widget_r.setText('%.5f' % (r_coef**2))
        self._set_average()
        table.cellChanged.connect(self.cellChanged)

    def ignorePoint(self):
        'Ignore or unignore current point.'
        table = self.ui.table
        table.cellChanged.disconnect()
        for row in libqt.selected_rows(table):
            libqt.cross(libqt.iter_row(table, row=row))
        self._set_average()
        self._do_q()
        self.__tableheaderq()
        table.cellChanged.connect(self.cellChanged)

    def deletePoint(self):
        'Delete current row in the data table.'
        self.ui.table.cellChanged.disconnect()
        for row in reversed(sorted(libqt.selected_rows(self.ui.table))):
            self.ui.table.removeRow(row)
            del self.__rawdata[row]
        self._set_average()
        self._do_q()
        self.__tableheaderq()
        self.ui.table.cellChanged.connect(self.cellChanged)

    def addPoint(self):
        '''Add a new row to the main table.

        Add a new point below currently selected row or at the bottom
        if no row is selected.
        '''
        current_row = self.__common_addpoint1()
        self.__cutoff.insert(current_row + 1, None)
        self.__rawdata.insert(current_row + 1, None)
        # TODO add default values

        self.__common_addpoint2()

    def add_fromfile(self):
        """Import data from a file and make a new point from it.

        Pop up a dialog for import data and make a new point.
        """
        filters = "Agilent CSV (*.CSV)", "Cary CSV (*.CSV)", "All Files (*.*)"
        filename, ok = libqt.opendialog(parent=self, filters=";;".join(filters),
                                    directory=self.default_dir)
        if not ok:
            return

        fformat: int = filters.index(ok)
        if fformat > 1:
            fformat = 0

        try:
            data = libio.load_raw_spectrum(filename, csvformat=fformat)
        except Exception as e:
            msg = "Error while importing data from {}"
            libqt.popwarning(msg.format(filename), str(e))
            raise e

        bounds, slope = libsod.process_spectrum(data)
        current_row = self.__common_addpoint1(str(10000*slope))
        self.__cutoff.insert(current_row + 1, bounds)
        self.__rawdata.insert(current_row + 1, data.T)

        self.__common_addpoint2()

    def __common_addpoint1(self, slope='0.00'):
        self.ui.table.cellChanged.disconnect()
        current_row = self.ui.table.currentRow()
        self.ui.table.insertRow(current_row+1)

        for col, (txt, flag) in enumerate(((slope, CELL_EDITABLE),
                                           ('-', CELL_NONEDITABLE),
                                           ('-', CELL_NONEDITABLE))):
            qtwi = QWidgets.QTableWidgetItem(txt)
            qtwi.setFlags(flag)
            self.ui.table.setItem(1+current_row, col, qtwi)
        return current_row

    def __common_addpoint2(self):
        self._set_average()
        self.__tableheaderq()
        self._do_q()
        self.ui.table.cellChanged.connect(self.cellChanged)

    def editPoint(self):
        "Open data for this point in :class:`RawSpectrum`."
        with libqt.table_locked(self.ui.table) as table:
            curr = table.currentRow()
            data = self.data[curr]
            dialog = RawSpectrum(rawdata=data[2],
                                 lower_cutoff=data[1][0],
                                 upper_cutoff=data[1][1])
            retv = dialog.exec()
            if retv == QWidgets.QDialog.Accepted:
                slope, rsq, lowerc, upperc = dialog.result()
                self.data[curr] = (True, (lowerc, upperc), data[2])
                ws = table.item(curr, 0)
                ws.setText('%.5f' % (1e4 * slope))
                wr = table.item(curr, 1)
                wr.setText('%.5f' % rsq)
                self.__cutoff[curr] = (lowerc, upperc)
            self._do_q()
            self.__tableheaderq()
            self._set_average()


class RawSpectrum(QWidgets.QDialog):
    "Display and fit raw spectra."

    def __init__(self, rawdata, lower_cutoff, upper_cutoff):
        '''Initialize widget.

        Parameters:
            rawdata: the arrays containing the raw spectrum
            lower_cutoff: the lower bounds of the linear fit
            upper_cutoff: the upper bounds of the linear fit
        '''
        super().__init__()
        self.ui = importui.Ui_ImportData()
        self.ui.setupUi(self)

        self._rawdata = rawdata

        from matplotlib.backends.backend_qt5agg import FigureCanvas
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel(r'$t$ / s')
        self.axes.set_ylabel(r'Absorbance')
        self.ui.stackedWidget.addWidget(self.canvas)
        self.setWindowTitle('Linear Interval')

        self.ui.slider_initpoint.setMaximum(len(rawdata[0])-3)
        self.ui.slider_endpoint.setMaximum(len(rawdata[0])-1)
        self.ui.slider_initpoint.setValue(lower_cutoff)
        self.ui.lbl_initpoint.setNum(lower_cutoff)
        self.ui.slider_endpoint.setValue(upper_cutoff)
        self.ui.lbl_endpoint.setNum(upper_cutoff)
        self.ui.slider_endpoint.setSliderPosition(upper_cutoff)

        self.ui.slider_initpoint.sliderMoved.connect(self.linear_fit)
        self.ui.slider_endpoint.sliderMoved.connect(self.linear_fit)

        self.axes.scatter(self._rawdata[0], self._rawdata[1])

        self.fitline = self.axes.plot((rawdata[0, 0], rawdata[0, -1]),
                                      (rawdata[1, 0], rawdata[1, -1]), 'r-')[0]
        self.axes.autoscale(enable=True, axis='both', tight=True)
        self.linear_fit()

        self.ui.pb_accept.clicked.connect(super().accept)
        self.ui.pb_cancel.clicked.connect(super().reject)

    def linear_fit(self):
        "Perform linear fit and update plot."
        import scipy.stats as spst
        lower = self.ui.slider_initpoint.value()
        upper = self.ui.slider_endpoint.value()
        a_s, b_s, r_coef, _, _ = spst.linregress(
            self._rawdata[0, lower:upper],
            self._rawdata[1, lower:upper]
        )

        new_x = self._rawdata[0, (lower, upper)]
        new_y = (a_s * new_x[0] + b_s, a_s * new_x[1] + b_s)

        self._result = (a_s, r_coef**2, lower, upper)

        txt = "slope = {}, r<sup>2</sup> = {}"
        self.ui.lbl_output.setText(txt.format(a_s, r_coef**2))
        self.fitline.set_xdata(new_x)
        self.fitline.set_ydata(new_y)
        self.canvas.draw()

    def result(self):
        "Access to the result of the fitting."
        return self._result


def main():
    "Run application."
    app = QWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    QtCore.pyqtRemoveInputHook()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
