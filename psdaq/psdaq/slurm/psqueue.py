import sys
import os
import platform
import time
import getopt
import _thread
import locale
import traceback
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QVariant, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QCursor, QBrush, QColor
from PyQt5.QtWidgets import QApplication, QMessageBox, QTableWidgetItem
import subprocess


from psdaq.slurm.main import Runner
from psdaq.slurm import ui_procStat

__version__ = "0.6"

# OutDir Full Path Example: daq-sxr-ana01: /u2/pcds/pds/sxr/e19/e19-r0026-s00-c00.xtc
sOutDirPrefix1 = "/u2/pcds/pds/"
sOutFileExtension = ""
bProcMgrThreadError = False
sErrorOutDirNotExist = "Output Dir Not Exist"


class CustomIOError(Exception):
    pass


def printStackTrace():
    print("---- Printing program call stacks for debug ----")
    traceback.print_exc(file=sys.stdout)
    print("------------------------------------------------")
    return


def psbatchThreadWrapper(
    win,
    sConfigFile,
    fQueryInterval,
    evgProcMgr,
):
    try:
        psbatchThread(
            win,
            sConfigFile,
            fQueryInterval,
            evgProcMgr,
        )
    except:
        sErrorReport = (
            "psbatchThreadWrapper(): psbatchThread(ConfigFile = %s, Query Interval = %f ) Failed"
            % (sConfigFile, float(fQueryInterval))
        )
        print(sErrorReport)
        printStackTrace()
        bProcMgrThreadError = True
        win.UnknownError.emit(
            sErrorReport
        )  # Send out the signal to notify the main window
    return


def psbatchThread(
    win,
    sConfigFile,
    fQueryInterval,
    evgProcMgr,
):

    locale.setlocale(
        locale.LC_ALL, ""
    )  # set locale for printing formatted numbers later

    while True:
        # refresh ProcMgr status
        global bProcMgrThreadError

        try:
            runner = Runner(sConfigFile)
            win.runner = runner
            ldProcStatus = runner.show_status(quiet=True)
            win.ldProcStatus = ldProcStatus

        except IOError:
            ldProcStatus = list()  # default to empty list
            print("psbatchThread(): ProcMgr(%s): I/O error" % (sConfigFile))
            printStackTrace()
            # Send out the signal to notify the main window
            win.ProcMgrIOError.emit(sConfigFile)

        except:
            ldProcStatus = list()  # default to empty list
            print("psbatchThread(): ProcMgr(%s) Failed" % (sConfigFile))
            printStackTrace()
            # Send out the signal to notify the main window
            win.ProcMgrGeneralError.emit(sConfigFile)

        if True:
            # Send out the signal to notify the main window
            # TODO: We should only need to send ldProcStatus
            win.Updated.emit(ldProcStatus, [], 0, "None")
        time.sleep(fQueryInterval)

    return


class WinProcStat(QtWidgets.QMainWindow, ui_procStat.Ui_mainWindow):

    # define 'Updated' signal
    Updated = pyqtSignal(list, list, int, str)

    # define 'UnknownError' signal
    UnknownError = pyqtSignal(str)

    # define 'ProcMgrIOError' signal
    ProcMgrIOError = pyqtSignal(str, int)

    # define 'ProcMgrGeneralError' signal
    ProcMgrGeneralError = pyqtSignal(str, int)

    # define 'ThreadGeneralError' signal
    ThreadGeneralError = pyqtSignal(str, int)

    # define 'OutputDirError' signal
    OutputDirError = pyqtSignal(str)

    def __init__(self, evgProcMgr, parent=None):
        super(WinProcStat, self).__init__(parent)

        self.sCurKey = ""
        self.setupUi(self)

        # setup message box for displaying warning messages
        self.msgBox = QMessageBox(
            QMessageBox.Warning, "Warning", "", QMessageBox.Ok, self
        )
        self.msgBox.setWindowModality(Qt.NonModal)

        # adjust GUI settings
        self.bFirstSort = False

        # setup signal handlers

        # built-in signals
        self.pushButtonConsole.clicked.connect(self.onClickConsole)
        self.pushButtonLogfile.clicked.connect(self.onClickLogfile)
        self.pushButtonRestart.clicked.connect(self.onClickRestart)
        self.tableProcStat.cellClicked.connect(self.onProcCellClicked)

        # new signals defined using pyqtSignal()
        self.Updated.connect(self.onProcMgrUpdated)
        self.ProcMgrIOError.connect(self.onProcMgrIOError)
        self.ThreadGeneralError.connect(self.onThreadGeneralError)
        self.ProcMgrGeneralError.connect(self.onProcMgrGeneralError)
        self.OutputDirError.connect(self.onProcMgrOutputDirError)
        self.UnknownError.connect(self.onProcMgrUnknownError)

        self.iShowConsole = (
            0  # 0: Don't show console, 1: Show console, 2: Show logfile, 3: Restart
        )
        self.runner = None
        return

    def closeEvent(self, event):
        self.msgBox.close()
        return

    def onProcMgrUpdated(self, ldProcStatus, ldOutputFileStatus):
        self.statusbar.showMessage("Refreshing ProcMgr status...")

        self.tableProcStat.clear()
        self.tableProcStat.setSortingEnabled(False)
        self.tableProcStat.setRowCount(len(ldProcStatus))
        self.tableProcStat.setColumnCount(2)
        self.tableProcStat.setHorizontalHeaderLabels(["ID", "Status"])

        itemCur = None

        for iRow, dProcStatus in enumerate(ldProcStatus):

            # col 1 : UniqueID
            if isinstance(dProcStatus["showId"], str):
                # str
                showId = dProcStatus["showId"]
            else:
                # bytes
                showId = dProcStatus["showId"].decode()
            item = QTableWidgetItem(showId)
            item.setData(Qt.UserRole, QVariant(showId))
            self.tableProcStat.setItem(iRow, 0, item)

            if showId == self.sCurKey:
                itemCur = item

            # col 2 : Status
            sStatus = dProcStatus["status"]
            item = QTableWidgetItem()
            brush1 = QBrush(QColor.fromRgb(255, 255, 255))
            item.setForeground(brush1)
            bluColor = (0, 0, 192)
            grnColor = (0, 192, 0)
            ylwColor = (192, 192, 0)
            redColor = (255, 0, 0)
            statusColors = {
                "COMPLETED": bluColor,
                "COMPLETING": bluColor,
                "FAILED": redColor,
                "PENDING": ylwColor,
                "PREEMPTED": redColor,
                "RUNNING": grnColor,
                "SUSPENDED": redColor,
                "STOPPED": redColor,
            }
            item.setData(0, QVariant(sStatus))
            item.setBackground(QBrush(QColor.fromRgb(*statusColors[sStatus])))

            self.tableProcStat.setItem(iRow, 1, item)

        # end for iRow, key in enumerate( sorted(procMgr.d.iterkeys()) ):

        self.tableProcStat.setColumnWidth(0, 160)
        self.tableProcStat.setColumnWidth(1, 100)

        if not self.bFirstSort:
            self.bFirstSort = True
            self.tableProcStat.sortItems(1, Qt.AscendingOrder)

        self.tableProcStat.setSortingEnabled(True)
        if itemCur != None:
            self.tableProcStat.setCurrentItem(itemCur)

        self.statusbar.clearMessage()
        return

    @pyqtSlot(int, int)
    def onProcCellClicked(self, iRow, iCol):
        if self.runner == None:
            return
        if self.iShowConsole == 0:
            return

        item = self.tableProcStat.item(iRow, 0)
        showId = item.data(Qt.UserRole)
        if self.iShowConsole == 1:
            self.runner.spawnConsole(showId, self.ldProcStatus, False)
        elif self.iShowConsole == 2:
            self.runner.spawnLogfile(showId, self.ldProcStatus, False)
        elif self.iShowConsole == 3:
            # override cursor
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            self.runner.restart(unique_ids=showId)
            # restore cursor
            QApplication.restoreOverrideCursor()
        return

    def onClickConsole(self, bChecked):
        self.pushButtonLogfile.setChecked(False)
        self.pushButtonRestart.setChecked(False)
        if bChecked:
            self.iShowConsole = 1
        else:
            self.iShowConsole = 0

    def onClickLogfile(self, bChecked):
        self.pushButtonConsole.setChecked(False)
        self.pushButtonRestart.setChecked(False)
        if bChecked:
            self.iShowConsole = 2
        else:
            self.iShowConsole = 0

    def onClickRestart(self, bChecked):
        self.pushButtonConsole.setChecked(False)
        self.pushButtonLogfile.setChecked(False)
        if bChecked:
            self.iShowConsole = 3
        else:
            self.iShowConsole = 0

    @pyqtSlot(str, int)
    def onProcMgrIOError(self, sConfigFile):
        self.showWarningWindow(
            "IO Error",
            "<p>config file <b>%s</b> cannot be processed correctly, due to an IO Error.<p>"
            % (sConfigFile)
            + "<p>Please check the input file format, or specify a new config file instead.",
        )
        return

    @pyqtSlot(str, int)
    def onProcMgrGeneralError(self, sConfigFile):
        self.showWarningWindow(
            "General Error",
            "<p>config file <b>%s</b> cannot be processed correctly, due to a general Error.<p>"
            % (sConfigFile)
            + "<p>Please check the input file content, and the target machine status.",
        )
        return

    @pyqtSlot(str)
    def onProcMgrOutputDirError(self, sOutDirPrefix1):
        self.showWarningWindow(
            "Output Directory Does Not Exist",
            "<p>Please check if the system has access to output directory <i>%s</i>.<p>"
            % (sOutDirPrefix1),
        )
        return

    @pyqtSlot(str)
    def onThreadGeneralError(self, sErrorReport):
        self.showWarningWindow(
            "Thread General Error",
            "<p><i>%s</i><p>" % (sErrorReport)
            + "<p>thread had a general error. Please check the log file for more details.\n",
        )
        return

    @pyqtSlot(str)
    def onProcMgrUnknownError(self, sErrorReport):
        QMessageBox.critical(
            self,
            "Unknown Error",
            "<p><i>%s</i><p>" % (sErrorReport)
            + "<p>Not able to update the process status. Please check the log file for more details.\n",
        )
        self.close()
        return

    def showWarningWindow(self, title, text):
        self.msgBox.setWindowTitle(title)
        self.msgBox.setText(text)
        self.msgBox.show()
        return

    @pyqtSlot(int, int, int, int)
    def on_tableProcStat_currentCellChanged(self, iCurRow, iCurCol, iPrevRow, iPrevCol):
        if iCurRow < 0:
            return

        itemCur = self.tableProcStat.item(iCurRow, 0)
        if itemCur == None:
            return

        self.sCurKey = itemCur.data(Qt.UserRole)
        return

    @pyqtSlot()
    def on_actionOpen_triggered(self):
        sFnConfig = str(
            QFileDialog.getOpenFileName(self, "Config File", ".", "config files (*.py)")
        )
        return

    @pyqtSlot()
    def on_actionQuit_triggered(self):
        self.close()
        return

    @pyqtSlot()
    def on_actionAbout_triggered(self):
        QMessageBox.about(
            self,
            "About psqueue",
            """<b>Status Monitor</b> v %s
            <p>Copyright &copy; 2009 SLAC PCDS
            <p>This application is used to monitor daq process status and report output file status.
            <p>Python %s - Qt %s - PyQt %s on %s"""
            % (
                __version__,
                platform.python_version(),
                QT_VERSION_STR,
                PYQT_VERSION_STR,
                platform.system(),
            ),
        )
        return


def showUsage():
    print(
        """\
Usage: %s  [-i | --interval <Query Interval>]  <Config file>
  -i | --interval   <Query Interval>       Query interval in seconds (default: 5 seconds)

Program Version %s\
"""
        % (__file__, __version__)
    )
    return


def main():
    fProcmgrQueryInterval = 5.0

    (llsOptions, lsRemainder) = getopt.getopt(
        sys.argv[1:],
        "vh:i:",
        [
            "version",
            "help",
            "interval=",
        ],
    )

    for (sOpt, sArg) in llsOptions:
        if sOpt in ("-v", "-h", "--version", "--help"):
            showUsage()
            return 1
        elif sOpt in ("-i", "--interval"):
            fProcmgrQueryInterval = float(sArg)

    if len(lsRemainder) < 1:
        print(__file__ + ": Config file is not specified")
        showUsage()
        return 1

    sConfigFile = lsRemainder[0]

    evgProcMgr = QObject()

    app = QApplication([])
    app.setOrganizationName("SLAC")
    app.setOrganizationDomain("slac.stanford.edu")
    app.setApplicationName("psqueue")
    win = WinProcStat(evgProcMgr)
    win.show()

    _thread.start_new_thread(
        psbatchThreadWrapper,
        (
            win,
            sConfigFile,
            fProcmgrQueryInterval,
            evgProcMgr,
        ),
    )

    app.exec_()

    return


# Main Entry
def _do_main():
    iRet = 0

    try:
        iRet = main()
    except:
        iRet = 101
        print(__file__ + ": %s" % (sys.exc_info()[1]))
        print("---- Printing program call stacks for debug ----")
        traceback.print_exc(file=sys.stdout)
        print("------------------------------------------------")
        showUsage()

    sys.exit(iRet)


if __name__ == "__main__":
    _do_main()
