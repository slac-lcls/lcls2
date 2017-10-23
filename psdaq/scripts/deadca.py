import sys
import argparse
from psp import Pv
from PyQt4 import QtCore, QtGui
import time

NBeamSeq = 16

dstsel     = ['Include','DontCare']
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

class PvTxt(PvTextDisplay):

    def __init__(self, pv, label):
        super(PvTxt, self).__init__(label)
        self.connect_signal()

        self.pv = Pv.Pv(pv)
        self.pv.monitor_start()
        print 'Monitor started '+pv
        self.pv.add_monitor_callback(self.update)

    def update(self, err):
        print 'Update '+pv
        q = self.pv.value
        if err is None:
            s = QtCore.QString(q)
            self.valueSet.emit(s)
        else:
            print err

    def setPv(self):
        pass

class PvEditTxt(PvTextDisplay):

    def __init__(self, pv, label):
        super(PvEditTxt, self).__init__(label)
        self.connect_signal()
        self.editingFinished.connect(self.setPv)

        self.pv = Pv.Pv(pv)
        self.pv.monitor_start()
        print 'Monitor started '+pv
        self.pv.add_monitor_callback(self.update)

    def update(self, err):
        print 'Update '+pv
        q = self.pv.value
        if err is None:
            s = QtCore.QString(q)
            self.valueSet.emit(s)
        else:
            print err

    def setPv(self):
        pass

class PvEditInt(PvEditTxt):

    def __init__(self, pv, label):
        super(PvEditInt, self).__init__(pv, label)

    def setPv(self):
        value = self.text().toInt()
        self.pv.put(value)

    def update(self, err):
        print 'Update '+pv
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

class PvDblArrayW(QtGui.QLabel):

    valueSet = QtCore.pyqtSignal(QtCore.QString,name='valueSet')

    def __init__(self):
        super(PvDblArrayW, self).__init__('-')
        self.connect_signal()

    def connect_signal(self):
        self.valueSet.connect(self.setValue)

    def setValue(self,value):
        self.setText(value)

class PvDblArray:
    
    def __init__(self, pv, widgets):
        self.widgets = widgets
        self.pv = Pv.Pv(pv)
        self.pv.monitor_start()
        print 'Monitor started '+pv
        self.pv.add_monitor_callback(self.update)

    def update(self, err):
        q = self.pv.value
        if err is None:
            for i in range(len(q)):
                self.widgets[i].valueSet.emit(QtCore.QString.number(q[i],'f',4))
        else:
            print err

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

class PvDstTab(QtGui.QWidget):
    
    def __init__(self, pvname):
        super(PvDstTab,self).__init__()

        self.pv = Pv.Pv(pvname)

        self.chkBox = []
        layout = QtGui.QGridLayout()
        for i in range(NBeamSeq):
            layout.addWidget( QtGui.QLabel('D%d'%i), i/4, 2*(i%4) )
            chkB = QtGui.QCheckBox()
            layout.addWidget( chkB, i/4, 2*(i%4)+1 )
            chkB.clicked.connect(self.update)
            self.chkBox.append(chkB)
        self.setLayout(layout)

    def update(self):
        v = 0
        for i in range(NBeamSeq):
            if self.chkBox[i].isChecked():
                v |= (1<<i)
        self.pv.put(v)

class PvEditDst(QtGui.QWidget):
    
    def __init__(self, pvname, idx):
        super(PvEditDst, self).__init__()
        vbox = QtGui.QVBoxLayout()
        selcmb = PvEditCmb(pvname,dstsel)
        
        vbox.addWidget(selcmb)
        vbox.addWidget(PvDstTab(pvname+'_Mask'))
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
        #layout.addStretch
        if count == 1:
            layout.addWidget(widget(pvname, ''))
        else:
            for i in range(count):
                layout.addWidget(widget(pvname+'%d'%i, QtCore.QString.number(i)))
        #layout.addStretch
        parent.addLayout(layout)

def LblPushButton(parent, pvbase, name, count=1):
    return PvInput(PvPushButton, parent, pvbase, name, count)

def LblCheckBox(parent, pvbase, name, count=1):
    return PvInput(PvCheckBox, parent, pvbase, name, count)

def LblEditInt(parent, pvbase, name, count=1):
    return PvInput(PvEditInt, parent, pvbase, name, count)

def LblEditHML(parent, pvbase, name, count=1):
    return PvInput(PvEditHML, parent, pvbase, name, count)

def LblEditTS(parent, pvbase, name, count=1):
    return PvInput(PvEditTS, parent, pvbase, name, count)

def LblEditEvt(parent, pvbase, name, count=1):
    return PvInput(PvEditEvt, parent, pvbase, name, count)

def LblEditDst(parent, pvbase, name, count=1):
    return PvInput(PvEditDst, parent, pvbase, name, count)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, base, partn, shelf):
        MainWindow.setObjectName(QtCore.QString.fromUtf8("MainWindow"))
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        pvbase = base+':XPM:'+shelf+':'
        #ppvbase = pvbase+partn+':'
        ppvbase = base+':PART:'+partn+':'
        print 'pvbase : '+pvbase
        print 'ppvbase: '+ppvbase

        grid = QtGui.QGridLayout()

        textWidgets = []
        for i in range(32):
            textWidgets.append( PvDblArrayW() )
            
        # Need to wait for pv.get()
        time.sleep(2)

        for i in range(14):
            pv = Pv.Pv(pvbase+'LinkLabel%d'%i)
            grid.addWidget( QtGui.QLabel(pv.get()), i, 0 )
            grid.addWidget( textWidgets[i], i, 1 )

        for j in range(16,21):
            i = j-16
            pv = Pv.Pv(pvbase+'LinkLabel%d'%j)
            grid.addWidget( QtGui.QLabel(pv.get()), i, 2 )
            grid.addWidget( textWidgets[j], i, 3 )

        for j in range(28,32):
            i = j-22
            grid.addWidget( QtGui.QLabel('INH-%d'%(j-28)), i, 2 )
            grid.addWidget( textWidgets[j], i, 3 )

        self.deadflnk = PvDblArray( ppvbase+'DeadFLnk', textWidgets )

        self.centralWidget.setLayout(grid)
        self.centralWidget.resize(240,340)

        title = 'XPM:'+shelf+'\tPART:'+partn
        MainWindow.setWindowTitle(title)
        MainWindow.resize(240,340)
        MainWindow.setCentralWidget(self.centralWidget)

if __name__ == '__main__':
    print QtCore.PYQT_VERSION_STR

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("base", help="pv base to monitor", default="DAQ:LAB2")
    parser.add_argument("partition", help="partition to monitor")
    parser.add_argument("shelf", help="shelf to monitor")
    args = parser.parse_args()

    app = QtGui.QApplication([])
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.base,args.partition,args.shelf)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())
