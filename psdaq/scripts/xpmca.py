import sys
import argparse
from psp import Pv
from PyQt5 import QtCore, QtGui, QtWidgets

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


class PvDisplay(QtWidgets.QLabel):

    valueSet = QtCore.pyqtSignal('QString',name='valueSet')

    def __init__(self):
        QtWidgets.QLabel.__init__(self, "-")
        self.setMinimumWidth(100)

    def connect_signal(self):
        self.valueSet.connect(self.setValue)

    def setValue(self,value):
        self.setText(value)

class PvCString:
    def __init__(self, parent, pvbase, name, dName=None):
        layout = QtWidgets.QHBoxLayout()
        label  = QtWidgets.QLabel(name)
        label.setMinimumWidth(100)
        layout.addWidget(label)
        #layout.addStretch()
        self.__display = PvDisplay()
        self.__display.connect_signal()
        layout.addWidget(self.__display)
        parent.addLayout(layout)

        pvname = pvbase+name
        print(pvname)
        self.pv = Pv.Pv(pvname)
        self.pv.monitor_start()
        self.pv.add_monitor_callback(self.update)

    def update(self, err):
        q = self.pv.value
        if err is None:
            s = QString()
            slen = len(q)
            if slen > 64:
                slen = 64
            for i in range(slen):
                if q[i]==0:
                    break
                s.append(QChar(q[i]))
            self.__display.valueSet.emit(s)
        else:
            print(err)

class PvLabel:
    def __init__(self, parent, pvbase, name, dName=None, isInt=False):
        layout = QtWidgets.QHBoxLayout()
        label  = QtWidgets.QLabel(name)
        label.setMinimumWidth(100)
        layout.addWidget(label)
        #layout.addStretch()
        self.__display = PvDisplay()
        self.__display.connect_signal()
        layout.addWidget(self.__display)
        parent.addLayout(layout)

        pvname = pvbase+name
        print(pvname)
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
            s = QString('fail')
            try:
                if self.isInt:
                    s = QString("%s (0x%s)") % (QString(int(q)),QString(format(int(q), 'x')))
                    if dq is not None:
                        s = s + QString(" [%s (0x%s)]") % (QString(int(dq)), QString(format(int(dq), 'x')))
                else:
                    s = QString(q)
                    if dq is not None:
                        s = s + QString(" [%s]") % (QString(dq))
            except:
                v = ''
                for i in range(len(q)):
                    #v = v + ' %f'%q[i]
                    v = v + ' ' + QString(q[i])
                    if dq is not None:
                        v = v + QString(" [%s]") % (QString(dq[i]))
                        #v = v + ' [' + '%f'%dq[i] + ']'
                    if ((i%8)==7):
                        v = v + '\n'
                s = QString(v)

            self.__display.valueSet.emit(s)
        else:
            print(err)

class PvPushButton(QtWidgets.QPushButton):

    valueSet = QtCore.pyqtSignal('QString',name='valueSet')

    def __init__(self, pvname, label):
        super(PvPushButton, self).__init__(label)
        self.setMaximumWidth(25) # Revisit

        self.clicked.connect(self.buttonClicked)

        self.pv = Pv.Pv(pvname)

    def buttonClicked(self):
        self.pv.put(1)          # Value is immaterial

class CheckBox(QtWidgets.QCheckBox):

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
            print(err)

class PvTextDisplay(QtWidgets.QLineEdit):

    valueSet = QtCore.pyqtSignal('QString',name='valueSet')

    def __init__(self, label):
        super(PvTextDisplay, self).__init__("-")
        #self.setMinimumWidth(60)

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
        self.setMaximumWidth(70)

    def setPv(self):
        value = self.text().toInt()
        self.pv.put(value)

    def update(self, err):
        q = self.pv.value
        if err is None:
            s = QString('fail')
            try:
                s = QString("%s") % (QString(int(q)))
            except:
                v = ''
                for i in range(len(q)):
                    v = v + ' %f'%q[i]
                s = QString(v)

            self.valueSet.emit(s)
        else:
            print(err)


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
            print("Invalid character in string:", value)

    def update(self, err):
        q = self.pv.value
        if err is None:
            v = toLMH[q & 0x3]
            q >>= 2
            while q:
                v = toLMH[q & 0x3] + v
                q >>= 2
            s = QString(v)

            self.valueSet.emit(s)
        else:
            print(err)

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
            s = QString('fail')
            try:
                s = QString(q)
            except:
                v = ''
                for i in range(len(q)):
                    v = v + ' %f'%q[i]
                s = QString(v)

            self.valueSet.emit(s)
        else:
            print(err)

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
            print(err)


class PvCmb(PvEditCmb):

    def __init__(self, pvname, choices):
        super(PvCmb, self).__init__(pvname, choices)
        self.setEnabled(False)


class PvEvtTab(QtWidgets.QStackedWidget):

    def __init__(self, pvname, evtcmb):
        super(PvEvtTab,self).__init__()

        self.addWidget(PvEditCmb(pvname+'_FixedRate',fixedRates))

        acw = QtWidgets.QWidget()
        acl = QtWidgets.QVBoxLayout()
        acl.addWidget(PvEditCmb(pvname+'_ACRate',acRates))
        acl.addWidget(PvEditCmb(pvname+'_ACTimeslot',acTS))
        acw.setLayout(acl)
        self.addWidget(acw)

        sqw = QtWidgets.QWidget()
        sql = QtWidgets.QVBoxLayout()
        sql.addWidget(PvEditCmb(pvname+'_Sequence',seqIdxs))
        sql.addWidget(PvEditCmb(pvname+'_SeqBit',seqBits))
        sqw.setLayout(sql)
        self.addWidget(sqw)

        evtcmb.currentIndexChanged.connect(self.setCurrentIndex)

class PvEditEvt(QtWidgets.QWidget):

    def __init__(self, pvname, idx):
        super(PvEditEvt, self).__init__()
        vbox = QtWidgets.QVBoxLayout()
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
        print(pvname)

        layout = QtWidgets.QHBoxLayout()
        label  = QtWidgets.QLabel(name)
        label.setMinimumWidth(100)
        layout.addWidget(label)
        #layout.addStretch
        if count == 1:
            w = widget(pvname, '')
            w.setEnabled(enable)
            layout.addWidget(w)
        else:
            for i in range(count):
                w = widget(pvname+'%d'%(i+start), QString(i+istart))
                w.setEnabled(enable)
                layout.addWidget(w)
        #layout.addStretch
        parent.addLayout(layout)

def LblPushButton(parent, pvbase, name, count=1, start=0, istart=0):
    return PvInput(PvPushButton, parent, pvbase, name, count, start, istart)

def LblCheckBox(parent, pvbase, name, count=1, start=0, istart=0, enable=True):
    return PvInput(PvCheckBox, parent, pvbase, name, count, start, istart, enable)

def LblEditInt(parent, pvbase, name, count=1, start=0, istart=0, enable=True):
    return PvInput(PvEditInt, parent, pvbase, name, count, start, istart, enable)

def LblEditHML(parent, pvbase, name, count=1):
    return PvInput(PvEditHML, parent, pvbase, name, count)

def LblEditTS(parent, pvbase, name, count=1):
    return PvInput(PvEditTS, parent, pvbase, name, count)

def LblEditEvt(parent, pvbase, name, count=1):
    return PvInput(PvEditEvt, parent, pvbase, name, count)

def FrontPanelAMC(pvbase,iamc):
        dshbox = QtWidgets.QHBoxLayout()
        dsbox = QtWidgets.QGroupBox("Front Panel Links (AMC%d)"%iamc)
        dslo = QtWidgets.QVBoxLayout()
#        LblEditInt   (lol, pvbase, "LinkTxDelay",    NAmcs * NDsLinks)
#        LblEditInt   (lol, pvbase, "LinkPartition",  NAmcs * NDsLinks)
#        LblEditInt   (lol, pvbase, "LinkTrgSrc",     NAmcs * NDsLinks)
        LblPushButton(dslo, pvbase, "TxLinkReset",    NDsLinks, start=iamc*NDsLinks)
        LblPushButton(dslo, pvbase, "RxLinkReset",    NDsLinks, start=iamc*NDsLinks)
        LblPushButton(dslo, pvbase, "RxLinkDump" ,    NDsLinks, start=iamc*NDsLinks)
        LblCheckBox  (dslo, pvbase, "LinkEnable",     NDsLinks, start=iamc*NDsLinks)
        LblCheckBox  (dslo, pvbase, "LinkRxReady",    NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox  (dslo, pvbase, "LinkTxReady",    NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox  (dslo, pvbase, "LinkIsXpm",      NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox  (dslo, pvbase, "LinkLoopback",   NDsLinks)
#        LblCheckBox  (dslo, pvbase, "LinkRxErr",      NAmcs * NDsLinks, enable=False)
        LblEditInt   (dslo, pvbase, "LinkRxErr",      NDsLinks, start=iamc*NDsLinks, enable=False)
        LblEditInt   (dslo, pvbase, "LinkRxRcv",      NDsLinks, start=iamc*NDsLinks, enable=False)
        dsbox.setLayout(dslo)
        dshbox.addWidget(dsbox)
        dshbox.addStretch()
        return dshbox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, title):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        pvbase = title + ':'
        lol = QtWidgets.QVBoxLayout()
        lor = QtWidgets.QVBoxLayout()

        PvLabel(lol, pvbase, "PARTITIONS"  )
        PvLabel(lol, pvbase, "PAddr"       , isInt=True)
        PvCString(lol, pvbase, "FwBuild"     )

        LblPushButton(lol, pvbase, "ModuleInit"      )
        LblPushButton(lol, pvbase, "DumpPll",        NAmcs)
        LblPushButton(lol, pvbase, "DumpTiming",     2)

        LblPushButton(lol, pvbase, "ClearLinks"      )

        LblPushButton(lol, pvbase, "Inhibit"         )
        LblPushButton(lol, pvbase, "TagStream"       )

        lol.addLayout(FrontPanelAMC(pvbase,0))
        lol.addLayout(FrontPanelAMC(pvbase,1))

        bthbox = QtWidgets.QHBoxLayout()
        btbox = QtWidgets.QGroupBox("Backplane Tx Links")
        btlo = QtWidgets.QVBoxLayout()
        LblPushButton(btlo, pvbase, "TxLinkReset16",    1, 16, 0)
        LblCheckBox  (btlo, pvbase, "LinkTxReady16",    1, 16, 0, enable=False)
        btbox.setLayout(btlo)
        bthbox.addWidget(btbox)
        lol.addLayout(bthbox)

        bphbox = QtWidgets.QHBoxLayout()
        bpbox = QtWidgets.QGroupBox("Backplane Rx Links")
        bplo = QtWidgets.QVBoxLayout()
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
        LblEditInt   (bplo, pvbase, "LinkRxRcv",      5, 17, 3, enable=False)
        LblEditInt   (bplo, pvbase, "LinkRxErr",      5, 17, 3, enable=False)
        bpbox.setLayout(bplo)
        bphbox.addWidget(bpbox)
        bphbox.addStretch()
        lol.addLayout(bphbox)


        pllhbox = QtWidgets.QHBoxLayout()
        pllbox  = QtWidgets.QGroupBox("PLLs")
        pllvbox = QtWidgets.QVBoxLayout() 
        LblCheckBox  (pllvbox, pvbase, "PLL_LOS",        NAmcs, enable=False)
        LblCheckBox  (pllvbox, pvbase, "PLL_LOL",        NAmcs, enable=False)
        LblEditHML   (pllvbox, pvbase, "PLL_BW_Select",  NAmcs)
        LblEditHML   (pllvbox, pvbase, "PLL_FreqTable",  NAmcs)
        LblEditHML   (pllvbox, pvbase, "PLL_FreqSelect", NAmcs)
        LblEditHML   (pllvbox, pvbase, "PLL_Rate",       NAmcs)
        LblPushButton(pllvbox, pvbase, "PLL_PhaseInc",   NAmcs)
        LblPushButton(pllvbox, pvbase, "PLL_PhaseDec",   NAmcs)
        LblPushButton(pllvbox, pvbase, "PLL_Bypass",     NAmcs)
        LblPushButton(pllvbox, pvbase, "PLL_Reset",      NAmcs)
        LblPushButton(pllvbox, pvbase, "PLL_Skew",       NAmcs)
        pllbox.setLayout(pllvbox)
        pllhbox.addWidget(pllbox)
        pllhbox.addStretch()
        lol.addLayout(pllhbox)

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

        lor.addStretch()

        PvLabel(lor, pvbase, "BpClk"      )
        
        #lor.addStretch()

        ltable = QtWidgets.QWidget()
        ltable.setLayout(lol)
        rtable = QtWidgets.QWidget()
        rtable.setLayout(lor)

        lscroll = QtWidgets.QScrollArea()
        lscroll.setWidget(ltable)
        rscroll = QtWidgets.QScrollArea()
        rscroll.setWidget(rtable)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(lscroll)
        splitter.addWidget(rscroll)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(splitter)

        self.centralWidget.setLayout(layout)
        self.centralWidget.resize(940,840)

        MainWindow.resize(940,840)
        MainWindow.setWindowTitle(title)
        MainWindow.setCentralWidget(self.centralWidget)

if __name__ == '__main__':
    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("pv", help="pv to monitor")
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.pv)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())

