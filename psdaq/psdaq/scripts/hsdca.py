import sys
import argparse
import numpy
from psp import Pv
from PyQt5 import QtCore, QtGui, QtWidgets
from pvedit import *

NChannels = 4
NBuffers = 16

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

class PvArray(object):
    def __init__( self, pvname, Editable=True):
        self._display = []
        initPvMon(self,pvname)
        for i,v in enumerate(self.pv.value):
            lbl = QtWidgets.QLineEdit(str(v))
            lbl.setEnabled(Editable)
            lbl.editingFinished.connect(self.setPv)
            self._display.append(lbl)

    def setPv(self):
        try:
            value = numpy.arange(len(self._display))
            for i in range(len(self._display)):
                value[i] = int(self._display[i].text())
            self.pv.put(value)
        except:
            pass

    def update(self,err):
        q = list(self.pv.value)
        if err is None:
            try:
                for i in range(len(q)):
                    self._display[i].setText(str(q[i]))
            except:
                pass

class PvRow(PvArray):
    def __init__( self, grid, row, pvname, label, Editable=True):
        super(PvRow, self).__init__( pvname, Editable)

        grid.addWidget( QtWidgets.QLabel(label), row, 0, QtCore.Qt.AlignHCenter )
        for i in range(len(self.pv.value)):
            grid.addWidget( self._display[i], row, i+1, QtCore.Qt.AlignHCenter )
        
class PvCol(PvArray):
    def __init__( self, grid, col, pvname, label, Editable=True):
        super(PvCol, self).__init__( pvname, Editable)

        grid.addWidget( QtWidgets.QLabel(label), 0, col, QtCore.Qt.AlignHCenter )
        for i in range(len(self.pv.value)):
            grid.addWidget( self._display[i], i+1, col, QtCore.Qt.AlignHCenter )
        
class HsdConfig(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(HsdConfig, self).__init__()

        lo = QtWidgets.QVBoxLayout()
        glo = QtWidgets.QGridLayout()
        # Table headers
        glo.addWidget( QtWidgets.QLabel('Channel'), 0, 0,
                       QtCore.Qt.AlignHCenter )
        for i in range(NChannels):
            glo.addWidget( QtWidgets.QLabel('%d'%i), 0, i+1,
                           QtCore.Qt.AlignHCenter )
        # Table data
        PvRow( glo, 1, pvbase+':ENABLE'   , 'Enable')
        PvRow( glo, 2, pvbase+':RAW_GATE' , 'Raw Gate')
        PvRow( glo, 3, pvbase+':RAW_PS'   , 'Raw Prescale')
        PvRow( glo, 4, pvbase+':FEX_GATE' , 'Fex Gate')
        PvRow( glo, 5, pvbase+':FEX_PS'   , 'Fex Prescale')
        PvRow( glo, 6, pvbase+':FEX_YMIN' , 'Fex Ymin')
        PvRow( glo, 7, pvbase+':FEX_YMAX' , 'Fex Ymax')
        PvRow( glo, 8, pvbase+':FEX_XPRE' , 'Fex Xpre')
        PvRow( glo, 9, pvbase+':FEX_XPOST', 'Fex Xpost')
        lo.addLayout(glo)

        hlo = QtWidgets.QHBoxLayout()
        hlo.addWidget(QtWidgets.QLabel('Test Pattern (None=-1)'))
        hlo.addWidget(PvEditInt(pvbase+':TESTPATTERN',''))
        hlo.addStretch(1)
        lo.addLayout(hlo)

        lo.addWidget(PvPushButton( pvbase+':BASE:APPLYCONFIG', 'Configure'))
        lo.addWidget(PvPushButton( pvbase+':BASE:UNDOCONFIG' , 'Unconfigure'))

        lo.addStretch(1)
        
        self.setLayout(lo)

class HsdStatus(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(HsdStatus, self).__init__()

        prefix = pvbase+':'

        lo = QtWidgets.QVBoxLayout()

        PvLabel( lo, prefix, 'TIMFRAMECNT' )
        PvLabel( lo, prefix, 'TIMPAUSECNT' )

        glo = QtWidgets.QGridLayout()
        # Table headers
        glo.addWidget( QtWidgets.QLabel('Channel'), 0, 0,
                       QtCore.Qt.AlignHCenter )
        for i in range(NChannels):
            glo.addWidget( QtWidgets.QLabel('%d'%i), 0, i+1,
                           QtCore.Qt.AlignHCenter )
        # Table data
        PvRow( glo, 1, pvbase+':PGPLOCLINKRDY', 'PgpLocalRdy' , False)
        PvRow( glo, 2, pvbase+':PGPREMLINKRDY', 'PgpRemoteRdy' , False)
        PvRow( glo, 3, pvbase+':PGPTXCLKFREQ' , 'PgpTx clk freq' , False)
        PvRow( glo, 4, pvbase+':PGPRXCLKFREQ' , 'PgpRx clk freq' , False)
        PvRow( glo, 5, pvbase+':PGPTXCNT'   , 'PgpTx frame count' , False)
        PvRow( glo, 6, pvbase+':PGPTXERRCNT', 'PgpTx error count' , False)
        PvRow( glo, 7, pvbase+':PGPRXCNT'   , 'PgpRx opcode count', False)
        PvRow( glo, 8, pvbase+':PGPRXLAST'  , 'PgpRx last opcode' , False)
        PvRow( glo, 9, pvbase+':RAW_FREEBUFSZ'  , 'Raw Free Bytes' , False)
        PvRow( glo,10, pvbase+':RAW_FREEBUFEVT' , 'Raw Free Events', False)
        PvRow( glo,11, pvbase+':FEX_FREEBUFSZ'  , 'Fex Free Bytes' , False)
        PvRow( glo,12, pvbase+':FEX_FREEBUFEVT' , 'Fex Free Events', False)
        PvRow( glo,13, pvbase+':TESTPATTERR' , 'Test Pattern Errors', False)
        PvRow( glo,14, pvbase+':TESTPATTBIT' , 'Test Pattern ErrBits', False)
        PvRow( glo,15, pvbase+':WRFIFOCNT' , 'Write FIFO Count', False)
        PvRow( glo,16, pvbase+':RDFIFOCNT' , 'Read FIFO Count', False)

        lo.addLayout(glo)
        lo.addStretch(1)

        self.setLayout(lo)

class HsdDetail(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(HsdDetail, self).__init__()

        prefix = pvbase+':'

        lo = QtWidgets.QVBoxLayout()
        glo = QtWidgets.QGridLayout()
        # Table headers
        glo.addWidget( QtWidgets.QLabel('Buffer'), 0, 0,
                       QtCore.Qt.AlignHCenter )
        for i in range(NBuffers):
            glo.addWidget( QtWidgets.QLabel('%d'%i), i+1, 0,
                           QtCore.Qt.AlignHCenter )
        # Table data
        PvCol( glo, 1, pvbase+':RAW_BUFSTATE'   , 'Buffer State', False)
        PvCol( glo, 2, pvbase+':RAW_TRGSTATE'   , 'Trigger State', False)
        PvCol( glo, 3, pvbase+':RAW_BUFBEG'     , 'Begin Address', False)
        PvCol( glo, 4, pvbase+':RAW_BUFEND'     , 'End Address'  , False)
        lo.addLayout(glo)
        lo.addStretch(1)

        self.setLayout(lo)

class HsdExpert(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(HsdExpert, self).__init__()

        lo = QtWidgets.QVBoxLayout()
        
        lo.addWidget(PvPushButton( pvbase+':RESET', 'Reset'))
        lo.addWidget(PvCheckBox( pvbase+':PGPLOOPBACK', 'Loopback'))

        lo.addStretch(1)

        self.setLayout(lo)

class DetBase(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(DetBase, self).__init__()

        prefix = pvbase+':BASE:'
        lo = QtWidgets.QVBoxLayout()

        v = Pv.Pv(prefix+'INTTRIGRANGE').get()
        lo.addWidget(QtWidgets.QLabel('Internal Trigger Range: %u - %u clks'%(v[0],v[1])))

        v = Pv.Pv(prefix+'INTTRIGCLK').get()
        lo.addWidget(QtWidgets.QLabel('Internal Trigger Clock: %f MHz (%f ns)'%(v,1.e3/v)))

        v = Pv.Pv(prefix+'INTTRIGVAL').get()
        lo.addWidget(QtWidgets.QLabel('Internal Trigger Value: %u clks'%v))

        if True:
            hlo = QtWidgets.QHBoxLayout()
            hlo.addWidget(QtWidgets.QLabel('Absolute Trigger Target (usecs)'))
            hlo.addWidget(PvEditInt(prefix+'ABSTRIGTARGET',''))
            hlo.addStretch(1)
            lo.addLayout(hlo)

        v = Pv.Pv(prefix+'INTPIPEDEPTH').get()
        lo.addWidget(QtWidgets.QLabel('Internal Pipeline Depth: %u'%v))

        v = Pv.Pv(prefix+'INTAFULLVAL').get()
        lo.addWidget(QtWidgets.QLabel('Internal Almost Full Threshold: %u'%v))

        v = Pv.Pv(prefix+'MINL0INTERVAL').get()
        lo.addWidget(QtWidgets.QLabel('Minimum Trigger Spacing (usecs): %f'%v))

        v = Pv.Pv(prefix+'UPSTREAMRTT').get()
        lo.addWidget(QtWidgets.QLabel('Upstream round trip time (usecs): %f'%v))

        v = Pv.Pv(prefix+'DNSTREAMRTT').get()
        lo.addWidget(QtWidgets.QLabel('Downstream round trip time (usecs)): %f'%v))

#        PvLabel( lo, prefix, 'PARTITION' )
        lo.addWidget(PvEditInt( prefix+'PARTITION', 'Partition' ))

        lo.addStretch(1)

        self.setLayout(lo)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, title):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        maintab = QtWidgets.QTabWidget()
        maintab.addTab( HsdConfig(title), 'Config' )
        maintab.addTab( HsdStatus(title), 'Status' )
        maintab.addTab( HsdDetail(title), 'Detail' )
        maintab.addTab( HsdExpert(title), 'Expert' )
        maintab.addTab( DetBase  (title), 'Base' )

        lo = QtWidgets.QVBoxLayout()
        lo.addWidget(maintab)

        self.centralWidget.setLayout(lo)

        MainWindow.resize(500,550)
        MainWindow.setWindowTitle(title)
        MainWindow.setCentralWidget(self.centralWidget)

if __name__ == '__main__':
    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("base", help="pv base to monitor", default="DAQ:LAB2:HSD")
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.base)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())
