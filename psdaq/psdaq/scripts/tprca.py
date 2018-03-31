import sys
import argparse
from psp import Pv
from PyQt5 import QtCore, QtGui, QtWidgets

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
seqBits    = ['b%u'%i for i in range(32)]
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
        
        self.pv = Pv.Pv(pv)
        self.pv.monitor_start()
        self.pv.add_monitor_callback(self.update)

class PvEditInt(PvEditTxt):

    def __init__(self, pv):
        super(PvEditInt, self).__init__(pv)

    def setPv(self):
        value = int(self.text())
        self.pv.put(value)

    def update(self, err):
        q = self.pv.value
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

    def __init__(self, pv):
        super(PvEditDbl, self).__init__(pv)

    def setPv(self):
        value = float(self.text())
        self.pv.put(value)

    def update(self, err):
        q = self.pv.value
        if err is None:
            s = 'fail'
            try:
                s = '{:.2f}'.format(q)
            except:
                v = ''
                for i in range(len(q)):
                    v = v + ' {:.2f}'.format(q[i])
                s = v

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

        self.pv = Pv.Pv(pvname)

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

class PvRowDbl():
    def __init__(self, row, layout, prefix, pv, label, ncols=NReadoutChannels):
        qlabel = QtWidgets.QLabel(label)
        qlabel.setMinimumWidth(RowHdrLen)
        layout.addWidget(qlabel,row,0)

        for i in range(ncols):
            qedit = PvEditDbl(prefix+':CH%u:'%i+pv)
            layout.addWidget(qedit,row,i+1)
        row += 1

class PvRowInt():
    def __init__(self, row, layout, prefix, pv, label, ncols=NReadoutChannels):
        qlabel = QtWidgets.QLabel(label)
        qlabel.setMinimumWidth(RowHdrLen)
        layout.addWidget(qlabel,row,0)

        for i in range(ncols):
            qedit = PvEditInt(prefix+':CH%u:'%i+pv)
            layout.addWidget(qedit,row,i+1)
        row += 1

class PvRowCmb():
    def __init__(self, row, layout, prefix, pv, label, choices, ncols=NReadoutChannels):
        qlabel = QtWidgets.QLabel(label)
        qlabel.setMinimumWidth(RowHdrLen)
        layout.addWidget(qlabel,row,0)

        for i in range(ncols):
            qedit = PvEditCmb(prefix+':CH%u:'%i+pv, choices)
            layout.addWidget(qedit,row,i+1)
        row += 1

class PvRowMod():
    def __init__(self, row, layout, prefix, pv, label):
        qlabel = QtWidgets.QLabel(label)
        qlabel.setMinimumWidth(RowHdrLen)
        layout.addWidget(qlabel,row,0)

        for i in range(NReadoutChannels):
            qedit = PvEditCmb(prefix+':CH%u:'%i+pv, modes)
            layout.addWidget(qedit,row,i+1)
        row += 1

class PvRowEvt():
    def __init__(self, row, layout, prefix, ncols=NReadoutChannels):
        qlabel = QtWidgets.QLabel('Event')
        qlabel.setMinimumWidth(RowHdrLen)
        layout.addWidget(qlabel,row,0)

        for i in range(ncols):
            qedit = PvEditEvt(prefix+':CH%u:'%i)
            layout.addWidget(qedit,row,i+1)
        row += 1

class PvRowDst():
    def __init__(self, row, layout, prefix, ncols=NReadoutChannels):
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
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

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

        for i in range(NReadoutChannels):
            qlabel = QtWidgets.QLabel('CH%u'%i)
            layout.addWidget( qlabel, row+0, i+1, QtCore.Qt.AlignHCenter )
        PvRowMod( row+1, layout, pvname, "MODE" , "")
        PvRowDbl( row+2, layout, pvname, "DELAY", "Delay [sec]", NTriggerChannels)
        PvRowDbl( row+3, layout, pvname, "WIDTH", "Width [sec]", NTriggerChannels)
        PvRowCmb( row+4, layout, pvname, "POL"  , "Polarity", polarities, NTriggerChannels)
        PvRowDst( row+5, layout, pvname)
        PvRowEvt( row+6, layout, pvname)
        PvRowInt( row+7, layout, pvname, "BSTART", "BsaStart [pul]")
        PvRowInt( row+8, layout, pvname, "BWIDTH", "BsaWidth [pul]")
        PvRowDbl( row+9, layout, pvname, "RATE"  , "Rate [Hz]")
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(widget)
        lo = QtWidgets.QHBoxLayout()
        lo.addWidget(scroll)
        self.centralWidget.setLayout(lo)
        self.centralWidget.resize(1000,600)
        MainWindow.resize(1000,600)
            

if __name__ == '__main__':
    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("pv", help="pv to monitor")
    args = parser.parse_args()

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
