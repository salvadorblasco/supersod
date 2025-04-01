# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'widgets/importdialog.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ImportData(object):
    def setupUi(self, ImportData):
        ImportData.setObjectName("ImportData")
        ImportData.resize(536, 372)
        self.verticalLayout = QtWidgets.QVBoxLayout(ImportData)
        self.verticalLayout.setObjectName("verticalLayout")
        self.stackedWidget = QtWidgets.QStackedWidget(ImportData)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy)
        self.stackedWidget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.stackedWidget.setFrameShadow(QtWidgets.QFrame.Raised)
        self.stackedWidget.setObjectName("stackedWidget")
        self.verticalLayout.addWidget(self.stackedWidget)
        self.gridlayout = QtWidgets.QGridLayout()
        self.gridlayout.setObjectName("gridlayout")
        self.label = QtWidgets.QLabel(ImportData)
        self.label.setObjectName("label")
        self.gridlayout.addWidget(self.label, 0, 0, 1, 1)
        self.lbl_initpoint = QtWidgets.QLabel(ImportData)
        self.lbl_initpoint.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lbl_initpoint.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lbl_initpoint.setObjectName("lbl_initpoint")
        self.gridlayout.addWidget(self.lbl_initpoint, 0, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(ImportData)
        self.label_2.setObjectName("label_2")
        self.gridlayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.lbl_endpoint = QtWidgets.QLabel(ImportData)
        self.lbl_endpoint.setObjectName("lbl_endpoint")
        self.gridlayout.addWidget(self.lbl_endpoint, 1, 2, 1, 1)
        self.slider_initpoint = QtWidgets.QSlider(ImportData)
        self.slider_initpoint.setOrientation(QtCore.Qt.Horizontal)
        self.slider_initpoint.setObjectName("slider_initpoint")
        self.gridlayout.addWidget(self.slider_initpoint, 0, 1, 1, 1)
        self.slider_endpoint = QtWidgets.QSlider(ImportData)
        self.slider_endpoint.setMinimum(2)
        self.slider_endpoint.setProperty("value", 99)
        self.slider_endpoint.setOrientation(QtCore.Qt.Horizontal)
        self.slider_endpoint.setObjectName("slider_endpoint")
        self.gridlayout.addWidget(self.slider_endpoint, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridlayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lbl_output = QtWidgets.QLabel(ImportData)
        self.lbl_output.setObjectName("lbl_output")
        self.horizontalLayout.addWidget(self.lbl_output)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.pb_accept = QtWidgets.QPushButton(ImportData)
        self.pb_accept.setObjectName("pb_accept")
        self.horizontalLayout.addWidget(self.pb_accept)
        self.pb_cancel = QtWidgets.QPushButton(ImportData)
        self.pb_cancel.setObjectName("pb_cancel")
        self.horizontalLayout.addWidget(self.pb_cancel)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(ImportData)
        self.slider_initpoint.sliderMoved['int'].connect(self.lbl_initpoint.setNum)
        self.slider_endpoint.sliderMoved['int'].connect(self.lbl_endpoint.setNum)
        QtCore.QMetaObject.connectSlotsByName(ImportData)

    def retranslateUi(self, ImportData):
        _translate = QtCore.QCoreApplication.translate
        ImportData.setWindowTitle(_translate("ImportData", "Import Data from CSV files"))
        self.label.setText(_translate("ImportData", "begin"))
        self.lbl_initpoint.setText(_translate("ImportData", "0"))
        self.label_2.setText(_translate("ImportData", "end"))
        self.lbl_endpoint.setText(_translate("ImportData", "99"))
        self.lbl_output.setText(_translate("ImportData", "TextLabel"))
        self.pb_accept.setText(_translate("ImportData", "Accept"))
        self.pb_cancel.setText(_translate("ImportData", "Cancel"))

