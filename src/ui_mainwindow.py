# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'widgets/mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(709, 363)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.table = QtWidgets.QTableWidget(self.centralwidget)
        self.table.setObjectName("table")
        self.table.setColumnCount(7)
        self.table.setRowCount(1)
        item = QtWidgets.QTableWidgetItem()
        self.table.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setItem(0, 5, item)
        self.verticalLayout.addWidget(self.table)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.cb_indicator = QtWidgets.QComboBox(self.centralwidget)
        self.cb_indicator.setObjectName("cb_indicator")
        self.cb_indicator.addItem("")
        self.cb_indicator.addItem("")
        self.horizontalLayout.addWidget(self.cb_indicator)
        self.inp_cind = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.inp_cind.sizePolicy().hasHeightForWidth())
        self.inp_cind.setSizePolicy(sizePolicy)
        self.inp_cind.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.inp_cind.setObjectName("inp_cind")
        self.horizontalLayout.addWidget(self.inp_cind)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.btn_fit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_fit.setObjectName("btn_fit")
        self.horizontalLayout.addWidget(self.btn_fit)
        self.verticalLayout.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 709, 23))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuFit = QtWidgets.QMenu(self.menubar)
        self.menuFit.setObjectName("menuFit")
        self.menuWeights = QtWidgets.QMenu(self.menuFit)
        self.menuWeights.setObjectName("menuWeights")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setEnabled(True)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNewProject = QtWidgets.QAction(MainWindow)
        self.actionNewProject.setObjectName("actionNewProject")
        self.actionOpenProject = QtWidgets.QAction(MainWindow)
        self.actionOpenProject.setObjectName("actionOpenProject")
        self.actionExportData = QtWidgets.QAction(MainWindow)
        self.actionExportData.setObjectName("actionExportData")
        self.actionExportFigure = QtWidgets.QAction(MainWindow)
        self.actionExportFigure.setObjectName("actionExportFigure")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionSaveProject = QtWidgets.QAction(MainWindow)
        self.actionSaveProject.setObjectName("actionSaveProject")
        self.actionPlotNormal = QtWidgets.QAction(MainWindow)
        self.actionPlotNormal.setCheckable(True)
        self.actionPlotNormal.setChecked(True)
        self.actionPlotNormal.setEnabled(False)
        self.actionPlotNormal.setObjectName("actionPlotNormal")
        self.actionPlotLinearized = QtWidgets.QAction(MainWindow)
        self.actionPlotLinearized.setCheckable(True)
        self.actionPlotLinearized.setObjectName("actionPlotLinearized")
        self.actionImportData = QtWidgets.QAction(MainWindow)
        self.actionImportData.setObjectName("actionImportData")
        self.actionWeightUnit = QtWidgets.QAction(MainWindow)
        self.actionWeightUnit.setObjectName("actionWeightUnit")
        self.actionWeightQuadratic = QtWidgets.QAction(MainWindow)
        self.actionWeightQuadratic.setObjectName("actionWeightQuadratic")
        self.actionWeightManual = QtWidgets.QAction(MainWindow)
        self.actionWeightManual.setObjectName("actionWeightManual")
        self.actionHelp_Menu = QtWidgets.QAction(MainWindow)
        self.actionHelp_Menu.setObjectName("actionHelp_Menu")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionFitReport = QtWidgets.QAction(MainWindow)
        self.actionFitReport.setObjectName("actionFitReport")
        self.actionSaveProjectAs = QtWidgets.QAction(MainWindow)
        self.actionSaveProjectAs.setObjectName("actionSaveProjectAs")
        self.actionAppendData = QtWidgets.QAction(MainWindow)
        self.actionAppendData.setObjectName("actionAppendData")
        self.menuFile.addAction(self.actionNewProject)
        self.menuFile.addAction(self.actionOpenProject)
        self.menuFile.addAction(self.actionSaveProject)
        self.menuFile.addAction(self.actionSaveProjectAs)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionAppendData)
        self.menuFile.addAction(self.actionImportData)
        self.menuFile.addAction(self.actionExportData)
        self.menuFile.addAction(self.actionExportFigure)
        self.menuFile.addAction(self.actionExit)
        self.menuWeights.addAction(self.actionWeightUnit)
        self.menuWeights.addAction(self.actionWeightQuadratic)
        self.menuFit.addAction(self.menuWeights.menuAction())
        self.menuFit.addAction(self.actionFitReport)
        self.menuHelp.addAction(self.actionHelp_Menu)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuFit.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.actionExit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SuperSOD, alpha version "))
        item = self.table.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", " 1 "))
        item = self.table.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "C / µM"))
        item = self.table.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "slope · s"))
        item = self.table.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "error · s"))
        item = self.table.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "IC / %"))
        item = self.table.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "error IC / %"))
        item = self.table.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "fit residual"))
        item = self.table.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "weight"))
        __sortingEnabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        item = self.table.item(0, 0)
        item.setText(_translate("MainWindow", "0"))
        item = self.table.item(0, 1)
        item.setText(_translate("MainWindow", "0"))
        item = self.table.item(0, 2)
        item.setText(_translate("MainWindow", "0"))
        item = self.table.item(0, 3)
        item.setText(_translate("MainWindow", "0"))
        item = self.table.item(0, 5)
        item.setText(_translate("MainWindow", "0"))
        self.table.setSortingEnabled(__sortingEnabled)
        self.label.setText(_translate("MainWindow", "Indicator"))
        self.cb_indicator.setItemText(0, _translate("MainWindow", "NBT"))
        self.cb_indicator.setItemText(1, _translate("MainWindow", "CYTC"))
        self.inp_cind.setText(_translate("MainWindow", "50"))
        self.label_2.setText(_translate("MainWindow", "µM"))
        self.btn_fit.setText(_translate("MainWindow", "Fit"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuFit.setTitle(_translate("MainWindow", "Fit"))
        self.menuWeights.setTitle(_translate("MainWindow", "weights"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionNewProject.setText(_translate("MainWindow", "New Project"))
        self.actionOpenProject.setText(_translate("MainWindow", "Open Project"))
        self.actionExportData.setText(_translate("MainWindow", "Export Data"))
        self.actionExportFigure.setText(_translate("MainWindow", "Export Figure"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionSaveProject.setText(_translate("MainWindow", "Save Project"))
        self.actionPlotNormal.setText(_translate("MainWindow", "Normal"))
        self.actionPlotLinearized.setText(_translate("MainWindow", "Linearized"))
        self.actionImportData.setText(_translate("MainWindow", "Import Data"))
        self.actionWeightUnit.setText(_translate("MainWindow", "unit"))
        self.actionWeightQuadratic.setText(_translate("MainWindow", "quadratic"))
        self.actionWeightManual.setText(_translate("MainWindow", "manual"))
        self.actionHelp_Menu.setText(_translate("MainWindow", "Help Menu"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionFitReport.setText(_translate("MainWindow", "report"))
        self.actionSaveProjectAs.setText(_translate("MainWindow", "Save Project as"))
        self.actionAppendData.setText(_translate("MainWindow", "Append Data"))

