# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'procStat.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(370, 1100)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.vboxlayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.vboxlayout.setContentsMargins(0, 5, 5, 0)
        self.vboxlayout.setSpacing(5)
        self.vboxlayout.setObjectName("vboxlayout")
        self.groupBoxProcessStatus = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(
            self.groupBoxProcessStatus.sizePolicy().hasHeightForWidth()
        )
        self.groupBoxProcessStatus.setSizePolicy(sizePolicy)
        self.groupBoxProcessStatus.setAlignment(QtCore.Qt.AlignLeading)
        self.groupBoxProcessStatus.setFlat(True)
        self.groupBoxProcessStatus.setCheckable(False)
        self.groupBoxProcessStatus.setObjectName("groupBoxProcessStatus")
        self.vboxlayout1 = QtWidgets.QVBoxLayout(self.groupBoxProcessStatus)
        self.vboxlayout1.setContentsMargins(0, 0, 0, 0)
        self.vboxlayout1.setSpacing(0)
        self.vboxlayout1.setObjectName("vboxlayout1")
        self.tableProcStat = QtWidgets.QTableWidget(self.groupBoxProcessStatus)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.tableProcStat.sizePolicy().hasHeightForWidth()
        )
        self.tableProcStat.setSizePolicy(sizePolicy)
        self.tableProcStat.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tableProcStat.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableProcStat.setAlternatingRowColors(True)
        self.tableProcStat.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.tableProcStat.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableProcStat.setShowGrid(False)
        self.tableProcStat.setColumnCount(0)
        self.tableProcStat.setObjectName("tableProcStat")
        self.tableProcStat.setRowCount(0)
        self.vboxlayout1.addWidget(self.tableProcStat)
        self.horizontalFrame = QtWidgets.QFrame(self.groupBoxProcessStatus)
        self.horizontalFrame.setObjectName("horizontalFrame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalFrame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButtonConsole = QtWidgets.QPushButton(self.horizontalFrame)
        self.pushButtonConsole.setCheckable(True)
        self.pushButtonConsole.setObjectName("pushButtonConsole")
        self.horizontalLayout.addWidget(self.pushButtonConsole)
        self.pushButtonLogfile = QtWidgets.QPushButton(self.horizontalFrame)
        self.pushButtonLogfile.setCheckable(True)
        self.pushButtonLogfile.setObjectName("pushButtonLogfile")
        self.horizontalLayout.addWidget(self.pushButtonLogfile)
        self.pushButtonRestart = QtWidgets.QPushButton(self.horizontalFrame)
        self.pushButtonRestart.setCheckable(True)
        self.pushButtonRestart.setObjectName("pushButtonRestart")
        self.horizontalLayout.addWidget(self.pushButtonRestart)
        self.vboxlayout1.addWidget(self.horizontalFrame)
        self.vboxlayout.addWidget(self.groupBoxProcessStatus)
        self.groupBoxOutputFileStatus = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(
            self.groupBoxOutputFileStatus.sizePolicy().hasHeightForWidth()
        )
        self.groupBoxOutputFileStatus.setSizePolicy(sizePolicy)
        self.groupBoxOutputFileStatus.setFlat(True)
        self.groupBoxOutputFileStatus.setObjectName("groupBoxOutputFileStatus")
        self.vboxlayout2 = QtWidgets.QVBoxLayout(self.groupBoxOutputFileStatus)
        self.vboxlayout2.setContentsMargins(0, 0, 0, 0)
        self.vboxlayout2.setSpacing(0)
        self.vboxlayout2.setObjectName("vboxlayout2")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBoxOutputFileStatus)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)
        self.textBrowser.setObjectName("textBrowser")
        self.vboxlayout2.addWidget(self.textBrowser)
        self.vboxlayout.addWidget(self.groupBoxOutputFileStatus)
        mainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "daqstat"))
        self.groupBoxProcessStatus.setTitle(_translate("mainWindow", "Process Status"))
        self.tableProcStat.setSortingEnabled(False)
        self.pushButtonConsole.setToolTip(
            _translate(
                "mainWindow",
                "Select this button, and click on processes to open consoles",
            )
        )
        self.pushButtonConsole.setText(_translate("mainWindow", "Open Console"))
        self.pushButtonLogfile.setToolTip(
            _translate(
                "mainWindow",
                "Select this button, and click on processes to view logfiles",
            )
        )
        self.pushButtonLogfile.setText(_translate("mainWindow", "View Logfile"))
        self.pushButtonRestart.setToolTip(
            _translate(
                "mainWindow",
                "Select this button, and click on remote processes to restart",
            )
        )
        self.pushButtonRestart.setText(_translate("mainWindow", "Restart"))
        self.groupBoxOutputFileStatus.setTitle(
            _translate("mainWindow", "Output File Status")
        )
