import sys
import argparse
from psp import Pv
from PyQt4 import QtCore, QtGui

NDsLinks    = 7
NAmcs       = 2
NPartitions = 16

evtsel      = ['Fixed Rate','AC Rate','Sequence']
fixedRates  = ['929kHz','71.4kHz','10.2kHz','1.02kHz','102Hz','10.2Hz','1.02Hz']
acRates     = ['60Hz','30Hz','10Hz','5Hz','1Hz']
acTS        = ['TS%u'%(i+1) for i in range(6)]
seqIdxs     = ['s%u'%i for i in range(18)]
seqBits     = ['b%u'%i for i in range(32)]


class PvDisplay(QtGui.QLabel):

    valueSet = QtCore.pyqtSignal(QtCore.QString,name='valueSet')

    def __init__(self):
        QtGui.QLabel.__init__(self, "-")
        self.setMinimumWidth(100)

    def connect_signal(self):
        self.valueSet.connect(self.setValue)

    def setValue(self,value):
        self.setText(value)

class PvLabel:
    def __init__(self, parent, pvbase, name, dName=None, isInt=False):
        layout = QtGui.QHBoxLayout()
        label  = QtGui.QLabel(name)
        label.setMinimumWidth(100)
#       label.setMinimumWidth(300)
        layout.addWidget(label)
        #layout.addStretch()
        self.__display = PvDisplay()
        self.__display.connect_signal()
        self.__display.setMinimumWidth(300)
        layout.addWidget(self.__display)
        parent.addLayout(layout)

        pvname = pvbase+name
        print pvname
        self.pv = Pv.Pv(pvname)
        self.pv.monitor_start()
        self.pv.add_monitor_callback(self.update)
        if dName is not None:
            dPvName = pvbase+dName
            self.dPv = Pv.Pv(dPvName)
            self.dPv.monitor_start()
            self.dPv.add_monitor_callback(self.update)
        else:
            self.dPv = None
        self.isInt = isInt

    def update(self, err):
        q = self.pv.value
        if self.dPv is not None:
            dq = self.dPv.value
        else:
            dq = None
        if err is None:
            s = QtCore.QString('fail')
            try:
                if self.isInt:
                    s = QtCore.QString("%1 (0x%2)").arg(QtCore.QString.number(long(q),10)).arg(QtCore.QString.number(long(q),16))
                    if dq is not None:
                        s = s + QtCore.QString(" [%1 (0x%2)]").arg(QtCore.QString.number(long(dq),10)).arg(QtCore.QString.number(long(dq),16))
                else:
                    s = QtCore.QString.number(q)
                    if dq is not None:
                        s = s + QtCore.QString(" [%1]").arg(QtCore.QString.number(dq))
            except:
                v = ''
                for i in range(len(q)):
                    #v = v + ' %f'%q[i]
                    v = v + ' ' + QtCore.QString.number(q[i])
                    if dq is not None:
                        v = v + QtCore.QString(" [%1]").arg(QtCore.QString.number(dq[i]))
                        #v = v + ' [' + '%f'%dq[i] + ']'
                    if ((i%8)==7):
                        v = v + '\n'
                s = QtCore.QString(v)

            self.__display.valueSet.emit(s)
            self.adjustSize()
        else:
            print err

class PvPushButton(QtGui.QPushButton):

    valueSet = QtCore.pyqtSignal(QtCore.QString,name='valueSet')

    def __init__(self, pvname, label):
        super(PvPushButton, self).__init__(label)
        self.setMaximumWidth(25) # Revisit

        self.clicked.connect(self.buttonClicked)

        self.pv = Pv.Pv(pvname)

    def buttonClicked(self):
        self.pv.put(1)          # Value is immaterial

class CheckBox(QtGui.QCheckBox):

    valueSet = QtCore.pyqtSignal(int, name='valueSet')

    def __init__(self, label):
        super(CheckBox, self).__init__(label)

    def connect_signal(self):
        self.valueSet.connect(self.boxClicked)

    def boxClicked(self, state):
        #print "CheckBox.clicked: state:", state
        self.setChecked(state)

class PvCheckBox(CheckBox):

    def __init__(self, pvname, label):
        super(PvCheckBox, self).__init__(label)
        self.connect_signal()
        self.clicked.connect(self.pvClicked)

        self.pv = Pv.Pv(pvname)
        self.pv.monitor_start()
        self.pv.add_monitor_callback(self.update)

    def pvClicked(self):
        q = self.isChecked()
        self.pv.put(q)
        #print "PvCheckBox.clicked: pv %s q %x" % (self.pv.name, q)

    def update(self, err):
        #print "PvCheckBox.update:  pv %s, i %s, v %x, err %s" % (self.pv.name, self.text(), self.pv.value, err)
        q = self.pv.value != 0
        if err is None:
            if q != self.isChecked():  self.valueSet.emit(q)
        else:
            print err

class PvTextDisplay(QtGui.QLineEdit):

    valueSet = QtCore.pyqtSignal(QtCore.QString,name='valueSet')

    def __init__(self, label):
        super(PvTextDisplay, self).__init__("-")
        #self.setMinimumWidth(60)

    def connect_signal(self):
        self.valueSet.connect(self.setValue)

    def setValue(self,value):
        self.setText(value)

class PvComboDisplay(QtGui.QComboBox):

    valueSet = QtCore.pyqtSignal(QtCore.QString,name='valueSet')

    def __init__(self, choices):
        super(PvComboDisplay, self).__init__()
        self.addItems(choices)

    def connect_signal(self):
        self.valueSet.connect(self.setValue)

    def setValue(self,value):
        self.setCurrentIndex(value)

class PvEditTxt(PvTextDisplay):

    def __init__(self, pv, label):
        super(PvEditTxt, self).__init__(label)
        self.connect_signal()
        self.editingFinished.connect(self.setPv)

        self.pv = Pv.Pv(pv)
        self.pv.monitor_start()
        self.pv.add_monitor_callback(self.update)

class PvEditInt(PvEditTxt):

    def __init__(self, pv, label):
        super(PvEditInt, self).__init__(pv, label)

    def setPv(self):
        value = self.text().toInt()
        self.pv.put(value)

    def update(self, err):
        q = self.pv.value
        if err is None:
            s = QtCore.QString('fail')
            try:
                s = QtCore.QString("%1").arg(QtCore.QString.number(long(q),10))
            except:
                v = ''
                for i in range(len(q)):
                    v = v + ' %f'%q[i]
                s = QtCore.QString(v)

            self.valueSet.emit(s)
        else:
            print err


class PvInt(PvEditInt):

    def __init__(self,pv):
        super(PvInt, self).__init__(pv)
        self.setEnabled(False)

class PvEditDbl(PvEditTxt):

    def __init__(self, pv, label):
        super(PvEditDbl, self).__init__(pv, label)

    def setPv(self):
        value = self.text().toDouble()
        self.pv.put(value)

    def update(self, err):
        q = self.pv.value
        if err is None:
            s = QtCore.QString('fail')
            try:
                s = QtCore.QString.number(q)
            except:
                v = ''
                for i in range(len(q)):
                    v = v + ' %f'%q[i]
                s = QtCore.QString(v)

            self.valueSet.emit(s)
        else:
            print err

class PvDbl(PvEditDbl):

    def __init__(self,pv):
        super(PvDbl, self).__init__(pv)
        self.setEnabled(False)


class PvEditCmb(PvComboDisplay):

    def __init__(self, pvname, choices):
        super(PvEditCmb, self).__init__(choices)
        self.connect_signal()
        self.currentIndexChanged.connect(self.setValue)

        self.pv = Pv.Pv(pvname)
        self.pv.monitor_start()
        self.pv.add_monitor_callback(self.update)

    def setValue(self):
        value = self.currentIndex()
        self.pv.put(value)

    def update(self, err):
        q = self.pv.value
        if err is None:
            self.setCurrentIndex(q)
            self.valueSet.emit(q)
        else:
            print err


class PvCmb(PvEditCmb):

    def __init__(self, pvname, choices):
        super(PvCmb, self).__init__(pvname, choices)
        self.setEnabled(False)


class PvEvtTab(QtGui.QStackedWidget):

    def __init__(self, pvname, evtcmb):
        super(PvEvtTab,self).__init__()

        self.addWidget(PvEditCmb(pvname+'_FixedRate',fixedRates))

        acw = QtGui.QWidget()
        acl = QtGui.QVBoxLayout()
        acl.addWidget(PvEditCmb(pvname+'_ACRate',acRates))
        acl.addWidget(PvEditCmb(pvname+'_ACTimeslot',acTS))
        acw.setLayout(acl)
        self.addWidget(acw)

        sqw = QtGui.QWidget()
        sql = QtGui.QVBoxLayout()
        sql.addWidget(PvEditCmb(pvname+'_Sequence',seqIdxs))
        sql.addWidget(PvEditCmb(pvname+'_SeqBit',seqBits))
        sqw.setLayout(sql)
        self.addWidget(sqw)

        evtcmb.currentIndexChanged.connect(self.setCurrentIndex)

class PvEditEvt(QtGui.QWidget):

    def __init__(self, pvname, idx):
        super(PvEditEvt, self).__init__()
        vbox = QtGui.QVBoxLayout()
        evtcmb = PvEditCmb(pvname,evtsel)
        vbox.addWidget(evtcmb)
        vbox.addWidget(PvEvtTab(pvname,evtcmb))
        self.setLayout(vbox)

class PvEditTS(PvEditCmb):

    def __init__(self, pvname, idx):
        super(PvEditTS, self).__init__(pvname, ['%u'%i for i in range(16)])

class PvInput:
    def __init__(self, widget, parent, pvbase, name, count=1):
        pvname = pvbase+name
        print pvname

        layout = QtGui.QHBoxLayout()
        label  = QtGui.QLabel(name)
        label.setMinimumWidth(100)
        layout.addWidget(label)
        if count == 1:
            # left justify using addStretch()
            layout.addWidget(widget(pvname, ''))
            layout.addStretch(1)
        else:
            for i in range(count):
                layout.addWidget(widget(pvname+'%d'%i, QtCore.QString.number(i)))
        parent.addLayout(layout)

def LblPushButton(parent, pvbase, name, count=1):
    return PvInput(PvPushButton, parent, pvbase, name, count)

def LblCheckBox(parent, pvbase, name, count=1):
    return PvInput(PvCheckBox, parent, pvbase, name, count)

def LblEditInt(parent, pvbase, name, count=1):
    return PvInput(PvEditInt, parent, pvbase, name, count)

def LblEditTS(parent, pvbase, name, count=1):
    return PvInput(PvEditTS, parent, pvbase, name, count)

def LblEditEvt(parent, pvbase, name, count=1):
    return PvInput(PvEditEvt, parent, pvbase, name, count)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, prefix):
        MainWindow.setObjectName(QtCore.QString.fromUtf8("MainWindow"))
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        lol = QtGui.QVBoxLayout()

        pvbase = prefix+':'

        LblPushButton(lol, pvbase, "CountClear"      )


        LblCheckBox  (lol, pvbase, "UsLinkEn",       7)

        LblEditInt   (lol, pvbase, "UsLinkPartition", 7)
        LblEditInt   (lol, pvbase, "UsLinkFwdMask",   7)
        LblEditInt   (lol, pvbase, "UsLinkTrigDelay", 7)

        lor = QtGui.QVBoxLayout()

        PvLabel(lor, pvbase, "UsLinkUp"  )
        PvLabel(lor, pvbase, "BpLinkUp"  )
        PvLabel(lor, pvbase, "DsLinkUp"  )

        PvLabel(lor, pvbase, "UsRxErrs", "dUsRxErrs"   )
        PvLabel(lor, pvbase, "UsRxFull", "dUsRxFull"   )
        PvLabel(lor, pvbase, "UsIbRecv", "dUsIbRecv"   )
        PvLabel(lor, pvbase, "UsIbEvt",  "dUsIbEvt"    )
        PvLabel(lor, pvbase, "DsRxErrs", "dDsRxErrs"   )
        PvLabel(lor, pvbase, "DsRxFull", "dDsRxFull"   )
        PvLabel(lor, pvbase, "DsObSent", "dDsObSent"   )

        PvLabel(lor, pvbase, "QpllLock"    )
        PvLabel(lor, pvbase, "MonClkRate"  )
        PvLabel(lor, pvbase, "MonClkSlow"  )
        PvLabel(lor, pvbase, "MonClkFast"  )
        PvLabel(lor, pvbase, "MonClkLock"  )

        PvLabel(lor, pvbase, "UsLinkObL0",  "dUsLinkObL0"    )
        PvLabel(lor, pvbase, "UsLinkObL1A", "dUsLinkObL1A"   )
        PvLabel(lor, pvbase, "UsLinkObL1R", "dUsLinkObL1R"   )
        PvLabel(lor, pvbase, "RxFrErrs", "dRxFrErrs"         )
        PvLabel(lor, pvbase, "RxFrames", "dRxFrames"         )
        PvLabel(lor, pvbase, "RxOpCodes", "dRxOpCodes"       )
        PvLabel(lor, pvbase, "TxFrErrs", "dTxFrErrs"         )
        PvLabel(lor, pvbase, "TxFrames", "dTxFrames"         )
        PvLabel(lor, pvbase, "TxOpCodes", "dTxOpCodes"       )

        ltable = QtGui.QWidget()
        ltable.setLayout(lol)
        rtable = QtGui.QWidget()
        rtable.setLayout(lor)

        lscroll = QtGui.QScrollArea()
        lscroll.setWidget(ltable)
        rscroll = QtGui.QScrollArea()
        rscroll.setWidget(rtable)

        splitter = QtGui.QSplitter()
        splitter.addWidget(lscroll)
        splitter.addWidget(rscroll)

        # splitter: left side stretch = 2x right side stretch
        splitter.setStretchFactor(0,2)
        splitter.setStretchFactor(1,1)

        layout = QtGui.QHBoxLayout()
        layout.addWidget(splitter)

        self.centralWidget.setLayout(layout)

        MainWindow.resize(1500,550)
        MainWindow.setWindowTitle(pvbase)

        MainWindow.setCentralWidget(self.centralWidget)

if __name__ == '__main__':
    print QtCore.PYQT_VERSION_STR

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("base", help="pv base to monitor", default="DAQ:LAB2:DTI")
    args = parser.parse_args()

    app = QtGui.QApplication([])
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.base)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())
