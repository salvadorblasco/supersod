# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'widgets/points.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PointDialog(object):
    def setupUi(self, PointDialog):
        PointDialog.setObjectName("PointDialog")
        PointDialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(PointDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.table = QtWidgets.QTableWidget(PointDialog)
        self.table.setObjectName("table")
        self.table.setColumnCount(3)
        self.table.setRowCount(1)
        item = QtWidgets.QTableWidgetItem()
        self.table.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(2, item)
        self.verticalLayout.addWidget(self.table)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lbl_mean = QtWidgets.QLabel(PointDialog)
        self.lbl_mean.setObjectName("lbl_mean")
        self.horizontalLayout.addWidget(self.lbl_mean)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.pb_accept = QtWidgets.QPushButton(PointDialog)
        self.pb_accept.setObjectName("pb_accept")
        self.horizontalLayout.addWidget(self.pb_accept)
        self.pb_cancel = QtWidgets.QPushButton(PointDialog)
        self.pb_cancel.setObjectName("pb_cancel")
        self.horizontalLayout.addWidget(self.pb_cancel)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(PointDialog)
        QtCore.QMetaObject.connectSlotsByName(PointDialog)

    def retranslateUi(self, PointDialog):
        _translate = QtCore.QCoreApplication.translate
        PointDialog.setWindowTitle(_translate("PointDialog", "Dialog"))
        item = self.table.verticalHeaderItem(0)
        item.setText(_translate("PointDialog", "1"))
        item = self.table.horizontalHeaderItem(0)
        item.setText(_translate("PointDialog", "slope"))
        item = self.table.horizontalHeaderItem(1)
        item.setText(_translate("PointDialog", "r²"))
        item = self.table.horizontalHeaderItem(2)
        item.setText(_translate("PointDialog", "Q"))
        self.lbl_mean.setText(_translate("PointDialog", "0.000+0.000"))
        self.pb_accept.setText(_translate("PointDialog", "Accept"))
        self.pb_cancel.setText(_translate("PointDialog", "Cancel"))

