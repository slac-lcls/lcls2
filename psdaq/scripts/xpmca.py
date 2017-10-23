import sys
import argparse
from psp import Pv
from PyQt4 import QtCore, QtGui

NDsLinks    = 7
NAmcs       = 2
NPartitions = 16

frLMH       = { 'L':0, 'H':1, 'M':2, 'm':3 }
toLMH       = { 0:'L', 1:'H', 2:'M', 3:'m' }

NBeamSeq = 16

dstsel     = ['DontCare','Exclude','Include']
bmsel      = ['D%u'%i for i in range(NBeamSeq)]
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

class PvCString:
    def __init__(self, parent, pvbase, name, dName=None):
        layout = QtGui.QHBoxLayout()
        label  = QtGui.QLabel(name)
        label.setMinimumWidth(100)
        layout.addWidget(label)
        #layout.addStretch()
        self.__display = PvDisplay()
        self.__display.connect_signal()
        layout.addWidget(self.__display)
        parent.addLayout(layout)

        pvname = pvbase+name
        print pvname
        self.pv = Pv.Pv(pvname)
        self.pv.monitor_start()
        self.pv.add_monitor_callback(self.update)

    def update(self, err):
        q = self.pv.value
        if err is None:
            s = QtCore.QString()
            for i in range(len(q)):
                if q[i]==0:
                    break
                s.append(QtCore.QChar(q[i]))
            self.__display.valueSet.emit(s)
        else:
            print err

class PvLabel:
    def __init__(self, parent, pvbase, name, dName=None, isInt=False):
        layout = QtGui.QHBoxLayout()
        label  = QtGui.QLabel(name)
        label.setMinimumWidth(100)
        layout.addWidget(label)
        #layout.addStretch()
        self.__display = PvDisplay()
        self.__display.connect_signal()
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

class PvEditHML(PvEditTxt):

    def __init__(self, pv, label):
        super(PvEditHML, self).__init__(pv, label)

    def setPv(self):
        value = self.text()
        try:
            q = 0
            for i in range(len(value)):
                q |= frLMH[str(value[i])] << (2 * (len(value) - 1 - i))
            self.pv.put(q)
        except KeyError:
            print "Invalid character in string:", value

    def update(self, err):
        q = self.pv.value
        if err is None:
            v = toLMH[q & 0x3]
            q >>= 2
            while q:
                v = toLMH[q & 0x3] + v
                q >>= 2
            s = QtCore.QString(v)

            self.valueSet.emit(s)
        else:
            print err

class PvHML(PvEditHML):

    def __init__(self, pv, label):
        super(PvHML, self).__init__(pv, label)
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
    def __init__(self, widget, parent, pvbase, name, count=1, start=0, istart=0, enable=True):
        pvname = pvbase+name
        print pvname

        layout = QtGui.QHBoxLayout()
        label  = QtGui.QLabel(name)
        label.setMinimumWidth(100)
        layout.addWidget(label)
        #layout.addStretch
        if count == 1:
            w = widget(pvname, '')
            w.setEnabled(enable)
            layout.addWidget(w)
        else:
            for i in range(count):
                w = widget(pvname+'%d'%(i+start), QtCore.QString.number(i+istart))
                w.setEnabled(enable)
                layout.addWidget(w)
        #layout.addStretch
        parent.addLayout(layout)

def LblPushButton(parent, pvbase, name, count=1, start=0, istart=0):
    return PvInput(PvPushButton, parent, pvbase, name, count, start, istart)

def LblCheckBox(parent, pvbase, name, count=1, start=0, istart=0, enable=True):
    return PvInput(PvCheckBox, parent, pvbase, name, count, start, istart, enable)

def LblEditInt(parent, pvbase, name, count=1):
    return PvInput(PvEditInt, parent, pvbase, name, count)

def LblEditHML(parent, pvbase, name, count=1):
    return PvInput(PvEditHML, parent, pvbase, name, count)

def LblEditTS(parent, pvbase, name, count=1):
    return PvInput(PvEditTS, parent, pvbase, name, count)

def LblEditEvt(parent, pvbase, name, count=1):
    return PvInput(PvEditEvt, parent, pvbase, name, count)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, title):
        MainWindow.setObjectName(QtCore.QString.fromUtf8("MainWindow"))
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        pvbase = title + ':'
        lol = QtGui.QVBoxLayout()
        lor = QtGui.QVBoxLayout()

        PvLabel(lol, pvbase, "PARTITIONS"  )
        PvLabel(lol, pvbase, "PAddr"       , isInt=True)
        PvCString(lol, pvbase, "FwBuild"     )

        LblPushButton(lol, pvbase, "ModuleInit"      )
        LblPushButton(lol, pvbase, "DumpPll",        NAmcs)
        LblPushButton(lol, pvbase, "DumpTiming",     2)

        LblPushButton(lol, pvbase, "ClearLinks"      )

        LblPushButton(lol, pvbase, "LinkDebug"       )
        LblPushButton(lol, pvbase, "Inhibit"         )
        LblPushButton(lol, pvbase, "TagStream"       )

        dsbox = QtGui.QGroupBox("Front Panel Links")
        dslo = QtGui.QVBoxLayout()
#        LblEditInt   (lol, pvbase, "LinkTxDelay",    NAmcs * NDsLinks)
#        LblEditInt   (lol, pvbase, "LinkPartition",  NAmcs * NDsLinks)
#        LblEditInt   (lol, pvbase, "LinkTrgSrc",     NAmcs * NDsLinks)
        LblPushButton(dslo, pvbase, "TxLinkReset",    NAmcs * NDsLinks)
        LblPushButton(dslo, pvbase, "RxLinkReset",    NAmcs * NDsLinks)
        LblCheckBox  (dslo, pvbase, "LinkEnable",     NAmcs * NDsLinks)
        LblCheckBox  (dslo, pvbase, "LinkRxReady",    NAmcs * NDsLinks, enable=False)
        LblCheckBox  (dslo, pvbase, "LinkTxReady",    NAmcs * NDsLinks, enable=False)
        LblCheckBox  (dslo, pvbase, "LinkIsXpm",      NAmcs * NDsLinks, enable=False)
        LblCheckBox  (dslo, pvbase, "LinkLoopback",   NAmcs * NDsLinks)
        LblCheckBox  (dslo, pvbase, "LinkRxErr",      NAmcs * NDsLinks, enable=False)
        dsbox.setLayout(dslo)
        lol.addWidget(dsbox)

        btbox = QtGui.QGroupBox("Backplane Tx Links")
        btlo = QtGui.QVBoxLayout()
        LblPushButton(btlo, pvbase, "TxLinkReset16",    1, 16, 0)
        LblCheckBox  (btlo, pvbase, "LinkTxReady16",    1, 16, 0, enable=False)
        btbox.setLayout(btlo)
        lol.addWidget(btbox)

        bpbox = QtGui.QGroupBox("Backplane Rx Links")
        bplo = QtGui.QVBoxLayout()
#        LblEditInt   (lol, pvbase, "LinkTxDelay",    5, 17, 3)
#        LblEditInt   (lol, pvbase, "LinkPartition",  5, 17, 3)
#        LblEditInt   (lol, pvbase, "LinkTrgSrc",     5, 17, 3)
#        LblPushButton(bplo, pvbase, "TxLinkReset",    5, 17, 3)
        LblPushButton(bplo, pvbase, "RxLinkReset",    5, 17, 3)
        LblCheckBox  (bplo, pvbase, "LinkEnable",     5, 17, 3)
        LblCheckBox  (bplo, pvbase, "LinkRxReady",    5, 17, 3, enable=False)
#        LblCheckBox  (bplo, pvbase, "LinkTxReady",    5, 17, 3, enable=False)
#        LblCheckBox  (bplo, pvbase, "LinkIsXpm",      5, 17, 3, enable=False)
#        LblCheckBox  (bplo, pvbase, "LinkLoopback",   5, 17, 3)
        LblCheckBox  (bplo, pvbase, "LinkRxErr",      5, 17, 3, enable=False)
        bpbox.setLayout(bplo)
        lol.addWidget(bpbox)

        LblCheckBox  (lol, pvbase, "PLL_LOS",        NAmcs, enable=False)
        LblCheckBox  (lol, pvbase, "PLL_LOL",        NAmcs, enable=False)
        LblEditHML   (lol, pvbase, "PLL_BW_Select",  NAmcs)
        LblEditHML   (lol, pvbase, "PLL_FreqTable",  NAmcs)
        LblEditHML   (lol, pvbase, "PLL_FreqSelect", NAmcs)
        LblEditHML   (lol, pvbase, "PLL_Rate",       NAmcs)
        LblPushButton(lol, pvbase, "PLL_PhaseInc",   NAmcs)
        LblPushButton(lol, pvbase, "PLL_PhaseDec",   NAmcs)
        LblPushButton(lol, pvbase, "PLL_Bypass",     NAmcs)
        LblPushButton(lol, pvbase, "PLL_Reset",      NAmcs)
        LblPushButton(lol, pvbase, "PLL_Skew",       NAmcs)

        if (False):
            LblEditEvt   (lol, pvbase, "L0Select"        )
            LblCheckBox  (lol, pvbase, "SetL0Enabled"    )

            LblCheckBox  (lol, pvbase, "L1TrgClear",     NPartitions)
            LblCheckBox  (lol, pvbase, "L1TrgEnable",    NPartitions)
            LblEditTS    (lol, pvbase, "L1TrgSource",    NPartitions)
            LblEditInt   (lol, pvbase, "L1TrgWord",      NPartitions)
            LblCheckBox  (lol, pvbase, "L1TrgWrite",     NPartitions)

            LblEditInt   (lol, pvbase, "AnaTagReset",    NPartitions)
            LblEditInt   (lol, pvbase, "AnaTag",         NPartitions)
            LblEditInt   (lol, pvbase, "AnaTagPush",     NPartitions)

            LblEditInt   (lol, pvbase, "PipelineDepth",  NPartitions)
            LblEditInt   (lol, pvbase, "MsgHeader",      NPartitions)
            LblCheckBox  (lol, pvbase, "MsgInsert",      NPartitions)
            LblEditInt   (lol, pvbase, "MsgPayload",     NPartitions)
            LblEditInt   (lol, pvbase, "InhInterval",    NPartitions)
            LblEditInt   (lol, pvbase, "InhLimit",       NPartitions)
            LblCheckBox  (lol, pvbase, "InhEnable",      NPartitions)

            #lol.addStretch()

            PvLabel(lor, pvbase, "L0InpRate"  )
            PvLabel(lor, pvbase, "L0AccRate"  )
            PvLabel(lor, pvbase, "L1Rate"     )
            PvLabel(lor, pvbase, "NumL0Inp"   )
            PvLabel(lor, pvbase, "NumL0Acc", None, True)
            PvLabel(lor, pvbase, "NumL1"      )
            PvLabel(lor, pvbase, "DeadFrac"   )
            PvLabel(lor, pvbase, "DeadTime"   )
            PvLabel(lor, pvbase, "DeadFLnk"   )

        PvLabel(lor, pvbase, "RxClks"     )
        PvLabel(lor, pvbase, "TxClks"     )
        PvLabel(lor, pvbase, "RxRsts"     )
        PvLabel(lor, pvbase, "CrcErrs"    )
        PvLabel(lor, pvbase, "RxDecErrs"  )
        PvLabel(lor, pvbase, "RxDspErrs"  )
        PvLabel(lor, pvbase, "BypassRsts" )
        PvLabel(lor, pvbase, "BypassDones")
        PvLabel(lor, pvbase, "RxLinkUp"   )
        PvLabel(lor, pvbase, "FIDs"       )
        PvLabel(lor, pvbase, "SOFs"       )
        PvLabel(lor, pvbase, "EOFs"       )
        
        #lor.addStretch()

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

        layout = QtGui.QHBoxLayout()
        layout.addWidget(splitter)

        self.centralWidget.setLayout(layout)
        self.centralWidget.resize(1040,840)

        MainWindow.resize(1040,840)
        MainWindow.setWindowTitle(title)
        MainWindow.setCentralWidget(self.centralWidget)

if __name__ == '__main__':
    print QtCore.PYQT_VERSION_STR

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("pv", help="pv to monitor")
    args = parser.parse_args()

    app = QtGui.QApplication([])
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.pv)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())
