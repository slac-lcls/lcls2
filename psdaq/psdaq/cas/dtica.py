import sys
import argparse
from PyQt5 import QtCore, QtGui, QtWidgets
from psdaq.cas.pvedit import *
from p4p.client.thread import Context

NUsLinks = 7
NDsLinks = 7
NPartitions = 8

try:
    QString = unicode
except NameError:
    # Python 3
    QString = str

try:
    QChar = unichr
except NameError:
    # Python 3
    QChar = chr

class PvPushButtonX(QtWidgets.QPushButton):

    valueSet = QtCore.pyqtSignal('QString',name='valueSet')

    def __init__(self, pvname, label):
        super(PvPushButtonX, self).__init__(label)
        self.setMaximumWidth(25) # Revisit

        self.clicked.connect(self.buttonClicked)

        self.pv = Pv(pvname, self.update)

    def update(self, err):
        pass

    def buttonClicked(self):
        self.pv.put(1)          # Value is immaterial

class PvEditIntX(PvEditInt):

    def __init__(self, pv, label):
        super(PvEditIntX, self).__init__(pv, label)
#       self.setMaximumWidth(70)

class PvEditCheckList:

    def __init__(self, pv, grid, row, col, entries):
        self.boxes = QtWidgets.QButtonGroup()
        for i in range(entries):
            cb = QtWidgets.QCheckBox()
            grid.addWidget( cb, row, col+i,
                            QtCore.Qt.AlignHCenter )
            self.boxes.addButton(cb,i)
        self.boxes.buttonClicked.connect(self.buttonClicked)
        initPvMon(self,pv)

    def buttonClicked(self, button):
        value = self.boxes.checkedId()-1
        self.pv.put(value)

    def update(self, err):
        q = self.pv.get()+1
        if err is None:
            try:
                self.boxes.button(int(q)).setChecked(True)
            except:
                pass
        else:
            print(err)

class PvCmb(PvEditCmb):

    def __init__(self, pvname, choices):
        super(PvCmb, self).__init__(pvname, choices)
        self.setEnabled(False)

def LblPushButtonX(parent, pvbase, name, count=1, start=0, istart=0):
    return PvInput(PvPushButtonX, parent, pvbase, name, count, start, istart)

def LblEditIntX(parent, pvbase, name, count=1, start=0, istart=0, enable=True):
    return PvInput(PvEditIntX, parent, pvbase, name, count, start, istart, enable)

class DtiAllocMon(object):
    def __init__(self, parent, pvname):
        self.parent = parent
        initPvMon(self,pvname)

    def update(self,err):
        self.parent.updateTable(err)

class DtiAllocation(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(DtiAllocation, self).__init__()

        glo = QtWidgets.QGridLayout()

        row = 0
        colp = 2
        cold = 12
        glo.addWidget( QtWidgets.QLabel('US'), row, 0,
                       QtCore.Qt.AlignHCenter )
        glo.addItem  ( QtWidgets.QSpacerItem( 15, 5 ), row, colp-1 )
        glo.addWidget( QtWidgets.QLabel('Partition'), row, colp, 1, 9,
                       QtCore.Qt.AlignHCenter )
        glo.addItem  ( QtWidgets.QSpacerItem( 15, 5 ), row, cold-1 )
        glo.addWidget( QtWidgets.QLabel('DS Links') , row, cold, 1, NDsLinks,
                       QtCore.Qt.AlignHCenter )
        row += 1
        glo.addWidget( QtWidgets.QLabel('None'), row, colp,
                       QtCore.Qt.AlignHCenter )
        for i in range(NPartitions):
            glo.addWidget( QtWidgets.QLabel('%d'%i), row, i+colp+1,
                           QtCore.Qt.AlignHCenter )
        for i in range(NUsLinks):
            glo.addWidget( QtWidgets.QLabel('%d'%i), row, i+cold,
                           QtCore.Qt.AlignHCenter )

        self.pvbase = pvbase
        self.groups = []
        for j in range(NDsLinks):
            self.groups.append(QtWidgets.QButtonGroup())
        for i in range(NUsLinks):
            row += 1
            glo.addWidget( QtWidgets.QLabel('%d'%i), row, 0,
                           QtCore.Qt.AlignHCenter )
            PvEditCheckList(pvbase+'UsLinkPartition%d'%i, glo, row, colp, 9)
            for j in range(NDsLinks):
                cb = QtWidgets.QCheckBox()
                self.groups[j].addButton(cb,i)
                glo.addWidget( cb, row, j+cold,
                               QtCore.Qt.AlignHCenter )
                cb.clicked.connect(self.update)

        self.mon = []
        for i in range(NUsLinks):
            self.mon.append(DtiAllocMon(self,pvbase+'UsLinkFwdMask%d'%i))

        self.setLayout(glo)

    def updateTable(self,err):
        usmask = [0]*len(self.mon)
        for i,mon in enumerate(self.mon):
            usmask[i] = mon.pv.get()
            for j in range(NDsLinks):
                if ((usmask[i] & (1<<j)) != 0):
                    self.groups[j].button(i).setChecked(True)

    def update(self):
        usmask = [0]*NUsLinks
        for j in range(NDsLinks):
            i = self.groups[j].checkedId()
            if i >= 0:
                usmask[i] = usmask[i] | (1<<j)
        for i in range(NUsLinks):
            self.mon[i].pv.put(usmask[i])

class DtiStatistics(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(DtiStatistics, self).__init__()
        self._pvlabels = []

        lor = QtWidgets.QVBoxLayout()
        if True:
            hbox = QtWidgets.QHBoxLayout()
            hbox.addLayout( LblMask(pvbase, 'BpLinkUp', 1) )
            hbox.addLayout( LblMask(pvbase, 'UsLinkUp', NUsLinks) )
            hbox.addLayout( LblMask(pvbase, 'DsLinkUp', NDsLinks) )
            lor.addLayout(hbox)

        lor.addWidget( PvPushButton(pvbase + "CountClear", "CountClear") )

        lor.addWidget(PvIntTable('Upstream Link Stats', pvbase,
                                 ['UsWrFifoD','UsRdFifoD','dUsIbEvt','UsObSent','UsObRecv','dUsRxFull','dUsRxInh','dUsRxErrs'],
                                 ['FifoWr'   ,'FifoRd'   ,'IbEvt'   ,'CtlOut'  ,'CtlIn'   ,'Full'     ,'InhEvts' ,'RxErrs'],
                                 NUsLinks))

        lor.addWidget(PvIntTable('Downstream Link Stats', pvbase,
                                 ['dDsRxErrs','dDsRxFull','dDsObSent'],
                                 ['RxErrs'   ,'Full' ,'MBytes'],
                                 NDsLinks))

        PvLabel(self, lor, pvbase, "QpllLock"    )
        PvLabel(self, lor, pvbase, "MonClkRate", scale=1.e-6, units='MHz' )
        PvLabel(self, lor, pvbase, "TimLinkUp"    )
        PvLabel(self, lor, pvbase, "TimRefClk" , scale=1.e-6, units='MHz' )
        PvLabel(self, lor, pvbase, "TimFrRate" , scale=1.e-3, units='kHz' )

        self.setLayout(lor)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, title):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        lol = QtWidgets.QVBoxLayout()
        lol.addWidget( QtWidgets.QLabel('Allocation') )

        alloctab = QtWidgets.QTabWidget()
        for i in range(3,8):
            pvslot = title + ':%d:'%i
            alloctab.addTab( DtiAllocation(pvslot), 'Slot-%d'%i )
        lol.addWidget(alloctab)

        lol.addWidget( QtWidgets.QLabel('Status') )

        statstab = QtWidgets.QTabWidget()
        for i in range(3,8):
            pvslot = title + ':%d:'%i
            statstab.addTab( DtiStatistics(pvslot), 'Slot-%d'%i )
        lol.addWidget(statstab)

        self.centralWidget.setLayout(lol)

        MainWindow.resize(500,550)
        MainWindow.setWindowTitle(title)
        MainWindow.setCentralWidget(self.centralWidget)

def main():
    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("base", help="pv base to monitor", default="DAQ:LAB2:DTI")
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.base)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
