import sys
import logging
import argparse
from PyQt5 import QtCore, QtGui, QtWidgets
from psdaq.cas.pvedit import Pv

logger = logging.getLogger(__name__)

NReadoutChannels = 14
NTriggerChannels = 12

accSel =     ['LCLS-I','LCLS-II']
linkStates = ['Down','Up']
rxpols     = ['Normal','Inverted']
RowHdrLen = 110
modes      = ['Disable','Trigger','+Readout','+BSA']
modesTTL   = ['Disable','Trigger']
polarities = ['Neg','Pos']
dstsel     = ['Any','Exclude','Include']
evtsel     = ['Fixed Rate','AC Rate','Sequence','Partition']
fixedRates = ['929kHz','71.4kHz','10.2kHz','1.02kHz','102Hz','10.2Hz','1.02Hz']
acRates    = ['60Hz','30Hz','10Hz','5Hz','1Hz']
acTS       = ['TS%u'%(i+1) for i in range(6)]
seqIdxs    = ['s%u'%i for i in range(18)]
seqBits    = ['b%u'%i for i in range(16)]
partitions = ['P%u'%i for i in range(8)]
ndestn     = 16

class PvTextDisplay(QtWidgets.QLineEdit):

    valueSet = QtCore.pyqtSignal('QString',name='valueSet')

    def __init__(self):
        super(PvTextDisplay, self).__init__("0")
        self.setMinimumWidth(60)

    def connect_signal(self):
        self.valueSet.connect(self.setValue)

    def setValue(self,value):
        self.setText(value)


class PvComboDisplay(QtWidgets.QComboBox):

    valueSet = QtCore.pyqtSignal('QString',name='valueSet')

    def __init__(self, choices):
        super(PvComboDisplay, self).__init__()
        self.addItems(choices)

    def connect_signal(self):
        self.valueSet.connect(self.setValue)

    def setValue(self,value):
        self.setCurrentIndex(value)

class PvEditTxt(PvTextDisplay):

    def __init__(self, pv):
        super(PvEditTxt, self).__init__()
        self.connect_signal()
        self.editingFinished.connect(self.setPv)

        self.pv = Pv(pv, self.update)

class PvEditInt(PvEditTxt):

    def __init__(self, pv):
        super(PvEditInt, self).__init__(pv)

    def setPv(self):
        value = int(self.text())
        self.pv.put(value)

    def update(self, err):
        q = self.pv.get()
        if err is None:
            s = 'fail'
            try:
                s = str(int(q))
            except:
                v = ''
                for i in range(len(q)):
                    v = v + ' %f'%q[i]
                s = v

            self.valueSet.emit(s)
        else:
            print(err)


class PvInt(PvEditInt):

    def __init__(self,pv):
        super(PvInt, self).__init__(pv)
        self.setEnabled(False)


class PvEditDbl(PvEditTxt):

    def __init__(self, pv,fmt='{:g}'):
        super(PvEditDbl, self).__init__(pv)
        self.fmt = fmt

    def setPv(self):
        value = float(self.text())
        self.pv.put(value)

    def update(self, err):
        q = self.pv.get()
        if err is None:
            s = 'fail'
            try:
                s = self.fmt.format(q)
            except:
                logger.exception("Excetion in pv edit double")
                v = ''
                for i in range(len(q)):
                    v = v + ' ' + self.fmt.format(q[i])
                s = v

            self.valueSet.emit(s)
        else:
            print(err)

class PvDbl(PvEditDbl):

    def __init__(self,pv,fmt='{:g}'):
        super(PvDbl, self).__init__(pv,fmt)
        self.setEnabled(False)


class PvEditCmb(PvComboDisplay):

    def __init__(self, pvname, choices):
        super(PvEditCmb, self).__init__(choices)
        self.connect_signal()
        self.currentIndexChanged.connect(self.setValue)

        self.pv = Pv(pvname, self.update)

    def setValue(self):
        value = self.currentIndex()
        if self.pv.get() != value:
            self.pv.put(value)
        else:
            logger.debug("Skipping updating PV for edit combobox as the value of the pv %s is the same as the current value", self.pv.pvname)

    def update(self, err):
        q = self.pv.get()
        if err is None:
            self.setCurrentIndex(q)
            self.valueSet.emit(str(q))
        else:
            print(err)


class PvCmb(PvEditCmb):

    def __init__(self, pvname, choices):
        super(PvCmb, self).__init__(pvname, choices)
        self.setEnabled(False)


class PvEvtTab(QtWidgets.QStackedWidget):

    def __init__(self, pvname, evtcmb):
        super(PvEvtTab,self).__init__()

        self.addWidget(PvEditCmb(pvname+'FRATE',fixedRates))

        acw = QtWidgets.QWidget()
        acl = QtWidgets.QVBoxLayout()
        acl.addWidget(PvEditCmb(pvname+'ARATE',acRates))
        acl.addWidget(PvEditCmb(pvname+'ATS'  ,acTS))
        acw.setLayout(acl)
        self.addWidget(acw)

        sqw = QtWidgets.QWidget()
        sql = QtWidgets.QVBoxLayout()
        sql.addWidget(PvEditCmb(pvname+'SEQIDX',seqIdxs))
        sql.addWidget(PvEditCmb(pvname+'SEQBIT',seqBits))
        sqw.setLayout(sql)
        self.addWidget(sqw)

        self.addWidget(PvEditCmb(pvname+'XPART',partitions))

        evtcmb.currentIndexChanged.connect(self.setCurrentIndex)

class PvEditEvt(QtWidgets.QWidget):

    def __init__(self, pvname):
        super(PvEditEvt, self).__init__()
        vbox = QtWidgets.QVBoxLayout()
        evtcmb = PvEditCmb(pvname+'RSEL',evtsel)
        vbox.addWidget(evtcmb)
        vbox.addWidget(PvEvtTab(pvname,evtcmb))
        self.setLayout(vbox)

class PvDstTab(QtWidgets.QWidget):

    def __init__(self, pvname):
        super(PvDstTab,self).__init__()

        self.pv = Pv(pvname)

        self.chkBox = []
        layout = QtWidgets.QGridLayout()
        for i in range(ndestn):
            layout.addWidget( QtWidgets.QLabel('D%d'%i), i/4, 2*(i%4) )
            chkB = QtWidgets.QCheckBox()
            layout.addWidget( chkB, i/4, 2*(i%4)+1 )
            chkB.clicked.connect(self.update)
            self.chkBox.append(chkB)
        self.setLayout(layout)

    def update(self):
        v = 0
        for i in range(ndestn):
            if self.chkBox[i].isChecked():
                v |= (1<<i)
        self.pv.put(v)

class PvEditDst(QtWidgets.QWidget):

    def __init__(self, pvname):
        super(PvEditDst, self).__init__()
        vbox = QtWidgets.QVBoxLayout()
        selcmb = PvEditCmb(pvname+'DSTSEL',dstsel)

        vbox.addWidget(selcmb)
        vbox.addWidget(PvDstTab(pvname+'DESTNS'))
        self.setLayout(vbox)

def PvRowDbl(row, layout, prefix, pv, label, ncols=NReadoutChannels, fmt='{}'):
    qlabel = QtWidgets.QLabel(label)
    qlabel.setMinimumWidth(RowHdrLen)
    layout.addWidget(qlabel,row,0)

    for i in range(ncols):
        qedit = PvEditDbl(prefix+':CH%u:'%i+pv, fmt)
        layout.addWidget(qedit,row,i+1)
    row += 1

def PvRowInt(row, layout, prefix, pv, label, ncols=NReadoutChannels):
    qlabel = QtWidgets.QLabel(label)
    qlabel.setMinimumWidth(RowHdrLen)
    layout.addWidget(qlabel,row,0)

    for i in range(ncols):
        qedit = PvEditInt(prefix+':CH%u:'%i+pv)
        layout.addWidget(qedit,row,i+1)
    row += 1

def PvRowCmb(row, layout, prefix, pv, label, choices, ncols=NReadoutChannels):
    qlabel = QtWidgets.QLabel(label)
    qlabel.setMinimumWidth(RowHdrLen)
    layout.addWidget(qlabel,row,0)

    for i in range(ncols):
        qedit = PvEditCmb(prefix+':CH%u:'%i+pv, choices)
        layout.addWidget(qedit,row,i+1)
    row += 1

def PvRowMod(row, layout, prefix, pv, label):
    qlabel = QtWidgets.QLabel(label)
    qlabel.setMinimumWidth(RowHdrLen)
    layout.addWidget(qlabel,row,0)

    for i in range(NReadoutChannels):
        qedit = PvEditCmb(prefix+':CH%u:'%i+pv, modes)
        layout.addWidget(qedit,row,i+1)
    row += 1

def PvRowEvt(row, layout, prefix, ncols=NReadoutChannels):
    qlabel = QtWidgets.QLabel('Event')
    qlabel.setMinimumWidth(RowHdrLen)
    layout.addWidget(qlabel,row,0)

    for i in range(ncols):
        qedit = PvEditEvt(prefix+':CH%u:'%i)
        layout.addWidget(qedit,row,i+1)
    row += 1

def PvRowDst(row, layout, prefix, ncols=NReadoutChannels):
    qlabel = QtWidgets.QLabel('Destn')
    qlabel.setMinimumWidth(RowHdrLen)
    layout.addWidget(qlabel,row,0)

    for i in range(ncols):
        qedit = PvEditDst(prefix+':CH%u:'%i)
        layout.addWidget(qedit,row,i+1)
    row += 1

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, pvname):
        MainWindow.setObjectName("MainWindow")

        tw = QtWidgets.QTabWidget(MainWindow)

        layout = QtWidgets.QGridLayout()

        row = 0
        layout.addWidget( QtWidgets.QLabel('ACCSEL'), row, 0 )
        layout.addWidget( PvEditCmb(pvname+':ACCSEL', accSel), row, 1 )
        row += 1
        layout.addWidget( QtWidgets.QLabel('LINKSTATE'), row, 0 )
        layout.addWidget( PvCmb(pvname+':LINKSTATE', linkStates), row, 1 )
        row += 1
        layout.addWidget( QtWidgets.QLabel('RXERRS'), row, 0 )
        layout.addWidget( PvInt(pvname+':RXERRS'), row, 1 )
        row += 1
        layout.addWidget( QtWidgets.QLabel('RXPOL'), row, 0 )
        layout.addWidget( PvEditCmb(pvname+':RXPOL', rxpols), row, 1 )
        row += 1
        layout.addWidget( QtWidgets.QLabel('FRAME RATE [Hz]'), row, 0 )
        layout.addWidget( PvDbl(pvname+':FRAMERATE'), row, 1 )
        row += 1
        layout.addWidget( QtWidgets.QLabel('RXCLK RATE [MHz]'), row, 0 )
        layout.addWidget( PvDbl(pvname+':RXCLKRATE'), row, 1 )
        row += 1
        layout.addWidget( QtWidgets.QLabel('IRQENA'), row, 0 )
        layout.addWidget( PvEditInt(pvname+':IRQENA'), row, 1 )
        row += 1
        layout.addWidget( QtWidgets.QLabel('EVTCNT'), row, 0 )
        layout.addWidget( PvEditInt(pvname+':EVTCNT'), row, 1 )
        row += 1
        layout.setColumnStretch(2,1)
        layout.setRowStretch(row,1)

        w = QtWidgets.QWidget()
        w.setLayout(layout)
        tw.addTab(w, 'Input')

        prefix = pvname
        for i in range(NReadoutChannels):
            lor = QtWidgets.QGridLayout()
            row = 0
            lor.addWidget(QtWidgets.QLabel('Event'),row,0)
            lor.addWidget(PvEditEvt(prefix+':CH%u:'%i),row,1)
            row += 1
            lor.addWidget(QtWidgets.QLabel('Destn'),row,0)
            lor.addWidget(PvEditDst(prefix+':CH%u:'%i),row,1)
            row += 1
            lor.addWidget(QtWidgets.QLabel('Rate'),row,0)
            lor.addWidget(PvEditDbl(prefix+':CH%u:RATE'%i, fmt='{:.2f}'),row,1)
            row += 1
            lor.addWidget(QtWidgets.QLabel('Mode'),row,0)
            lor.addWidget(PvEditCmb(prefix+':CH%u:MODE'%i, modes),row,1)
            row += 1
            lor.addWidget(QtWidgets.QLabel('Delay [sec]'),row,0)
            lor.addWidget(PvEditDbl(prefix+':CH%u:DELAY'%i),row,1)
            row += 1
            lor.addWidget(QtWidgets.QLabel('Width [sec]'),row,0)
            lor.addWidget(PvEditDbl(prefix+':CH%u:WIDTH'%i),row,1)
            row += 1
            lor.addWidget(QtWidgets.QLabel('Polarity'),row,0)
            lor.addWidget(PvEditCmb(prefix+':CH%u:POL'%i, polarities),row,1)
            w = QtWidgets.QWidget()
            w.setLayout(lor)
            tw.addTab( w, 'CH%u'%i )

        self.centralWidget = tw
        self.centralWidget.setObjectName("centralWidget")
        self.centralWidget.resize(600,600)
        MainWindow.resize(600,600)


def main():
    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    parser.add_argument("pv", help="pv to monitor")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    app = QtWidgets.QApplication([])
    #  Make disabled widgets just as visible as enabled widgets
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Disabled,
                     QtGui.QPalette.WindowText,
                     palette.color(QtGui.QPalette.Active,
                                   QtGui.QPalette.WindowText))
    palette.setBrush(QtGui.QPalette.Disabled,
                     QtGui.QPalette.WindowText,
                     palette.brush(QtGui.QPalette.Active,
                                   QtGui.QPalette.WindowText))
    app.setPalette(palette)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.pv)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
