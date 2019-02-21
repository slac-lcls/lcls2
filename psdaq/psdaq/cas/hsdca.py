import sys
import argparse
import logging
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from psdaq.cas.pvedit import *
from p4p.client.thread import Context

NChannels = 1
NLanes = 4
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
    def __init__( self, pvname, Editable=True, MaxLen=0):
        self._display = []
        initPvMon(self,pvname)
        dlen = len(self.pv.__value__)
        if (MaxLen>0 and MaxLen<dlen):
            print('Limiting %s to %d'%(pvname,MaxLen))
            dlen = MaxLen
        for i,v in enumerate(self.pv.__value__):
            if i<dlen:
                lbl = QtWidgets.QLineEdit(str(v))
                lbl.setEnabled(Editable)
                lbl.editingFinished.connect(self.setPv)
                self._display.append(lbl)

    def setPv(self):
        try:
            value = numpy.arange(len(self.pv.__value__))
            for i in range(len(self._display)):
                value[i] = int(self._display[i].text())
            self.pv.put(value)
        except:
            pass

    def update(self,err):
        if err is None:
            q = list(self.pv.__value__)
            try:
                for i in range(len(self._display)):
                    self._display[i].setText(str(q[i]))
            except:
                pass

class PvRow(PvArray):
    def __init__( self, grid, row, pvname, label, Editable=True, MaxLen=0 ):
        super(PvRow, self).__init__( pvname, Editable, MaxLen )

        grid.addWidget( QtWidgets.QLabel(label), row, 0, QtCore.Qt.AlignHCenter )
        for i in range(len(self._display)):
            grid.addWidget( self._display[i], row, i+1, QtCore.Qt.AlignHCenter )
        
class PvCol(PvArray):
    def __init__( self, grid, col, pvname, label, Editable=True):
        super(PvCol, self).__init__( pvname, Editable)

        grid.addWidget( QtWidgets.QLabel(label), 0, col, QtCore.Qt.AlignHCenter )
        for i in range(len(self._display)):
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
        pvtable = [('Enable'         ,'ENABLE'),
                   ('Raw Start'      ,'RAW_START'),
                   ('Raw Gate'       ,'RAW_GATE'),
                   ('Raw Prescale'   ,'RAW_PS'),
                   ('Fex Start'      ,'FEX_START'),
                   ('Fex Gate'       ,'FEX_GATE'),
                   ('Fex Prescale'   ,'FEX_PS'),
                   ('Fex Ymin'       ,'FEX_YMIN'),
                   ('Fex Ymax'       ,'FEX_YMAX'),
                   ('Fex Xpre'       ,'FEX_XPRE'),
                   ('Fex Xpost'      ,'FEX_XPOST')]
        for i,elem in enumerate(pvtable):
            PvRow( glo, i+1, pvbase+':'+elem[1], elem[0], MaxLen=NChannels )
        lo.addLayout(glo)

        pvtable = [('Test Pattern (None=-1)','TESTPATTERN')]
        for elem in pvtable:
            hlo = QtWidgets.QHBoxLayout()
            hlo.addWidget(QtWidgets.QLabel(elem[0]))
            hlo.addWidget(PvEditInt(pvbase+':'+elem[1],''))
            hlo.addStretch(1)
            lo.addLayout(hlo)

        lo.addWidget(PvPushButton ( pvbase+':BASE:APPLYCONFIG', 'Apply'))
        PvLabel      (lo, pvbase+':BASE:', 'READY'      , isInt=False)
#        lo.addWidget(PvPushButton( pvbase+':BASE:UNDOCONFIG' , 'Unconfigure'))
#        lo.addWidget(PvPushButton( pvbase+':BASE:ENABLETR'   , 'Enable'))
#        lo.addWidget(PvPushButton( pvbase+':BASE:DISABLETR'  , 'Disable'))

        lo.addStretch(1)
        
        self.setLayout(lo)

class HsdStatus(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(HsdStatus, self).__init__()

        prefix = pvbase+':'

        lo = QtWidgets.QVBoxLayout()

        PvLabel( lo, prefix, 'TIMFRAMECNT' )
        PvLabel( lo, prefix, 'TIMPAUSECNT' )
        PvLabel( lo, prefix, 'TRIGCNT' )
        PvLabel( lo, prefix, 'TRIGCNTSUM' )
#        PvLabel( lo, prefix, 'READCNTSUM' )
#        PvLabel( lo, prefix, 'STARTCNTSUM' )
        PvLabel( lo, prefix, 'MSGDELAYSET' )
        PvLabel( lo, prefix, 'MSGDELAYGET' )
        PvLabel( lo, prefix, 'HEADERCNTL0' )
        PvLabel( lo, prefix, 'HEADERCNTOF' )

        self._rows = []

        if True:
            glo = QtWidgets.QGridLayout()
            # Table headers
            glo.addWidget( QtWidgets.QLabel('Lane'), 0, 0,
                           QtCore.Qt.AlignHCenter )
            for i in range(NLanes):
                glo.addWidget( QtWidgets.QLabel('%d'%i), 0, i+1,
                               QtCore.Qt.AlignHCenter )
            # Table data
            self._rows.append(PvRow( glo, 1, pvbase+':PGPLOCLINKRDY', 'PgpLocalRdy' , False))
            self._rows.append(PvRow( glo, 2, pvbase+':PGPREMLINKRDY', 'PgpRemoteRdy' , False))
            self._rows.append(PvRow( glo, 3, pvbase+':PGPTXCLKFREQ' , 'PgpTx clk freq' , False))
            self._rows.append(PvRow( glo, 4, pvbase+':PGPRXCLKFREQ' , 'PgpRx clk freq' , False))
            self._rows.append(PvRow( glo, 5, pvbase+':PGPTXCNT'   , 'PgpTx frame rate' , False))
            self._rows.append(PvRow( glo, 6, pvbase+':PGPTXCNTSUM', 'PgpTx frame count' , False))
            self._rows.append(PvRow( glo, 7, pvbase+':PGPTXERRCNT', 'PgpTx error count' , False))
            self._rows.append(PvRow( glo, 8, pvbase+':PGPRXCNT'   , 'PgpRx opcode count', False))
            self._rows.append(PvRow( glo, 9, pvbase+':PGPRXLAST'  , 'PgpRx last opcode' , False))
            self._rows.append(PvRow( glo,10, pvbase+':PGPREMPAUSE'    , 'PgpRemPause/OF' , False))

            lo.addLayout(glo)
    
        if True:
            hb = QtWidgets.QHBoxLayout()
            if True:
                glo = QtWidgets.QGridLayout()
                self._rows.append(PvRow( glo,0, pvbase+':RAW_FREEBUFSZ'  , 'Raw Free Bytes' , False, MaxLen=NChannels))
                self._rows.append(PvRow( glo,1, pvbase+':RAW_FREEBUFEVT' , 'Raw Free Events', False, MaxLen=NChannels))
                self._rows.append(PvRow( glo,2, pvbase+':RAW_FIFOOF'     , 'Raw Fifo Overflow', False, MaxLen=NChannels))
                hb.addLayout(glo)

            if True:
                glo = QtWidgets.QGridLayout()
                self._rows.append(PvRow( glo,0, pvbase+':FEX_FREEBUFSZ'  , 'Fex Free Bytes' , False, MaxLen=NChannels))
                self._rows.append(PvRow( glo,1, pvbase+':FEX_FREEBUFEVT' , 'Fex Free Events', False, MaxLen=NChannels))
                self._rows.append(PvRow( glo,2, pvbase+':FEX_FIFOOF'     , 'Fex Fifo Overflow', False, MaxLen=NChannels))
                hb.addLayout(glo)
                lo.addLayout(hb)

            if True:
                glo = QtWidgets.QGridLayout()
                self._rows.append(PvRow( glo,0, pvbase+':DATA_FIFOOF'    , 'Data Fifo Overflow' , False, MaxLen=NChannels))
                self._rows.append(PvRow( glo,1, pvbase+':WRFIFOCNT' , 'Write FIFO Count', False, MaxLen=NChannels))
                self._rows.append(PvRow( glo,2, pvbase+':RDFIFOCNT' , 'Read FIFO Count', False, MaxLen=NChannels))
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
        self._cols = []
        self._cols.append(PvCol( glo, 1, pvbase+':RAW_BUFSTATE'   , 'Buffer State', False))
        self._cols.append(PvCol( glo, 2, pvbase+':RAW_TRGSTATE'   , 'Trigger State', False))
        self._cols.append(PvCol( glo, 3, pvbase+':RAW_BUFBEG'     , 'Begin Address', False))
        self._cols.append(PvCol( glo, 4, pvbase+':RAW_BUFEND'     , 'End Address'  , False))
        lo.addLayout(glo)
        lo.addStretch(1)

        PvLabel( lo, prefix, 'SYNCE' )
        PvLabel( lo, prefix, 'SYNCO' )
        lo.addStretch(1)

        self.setLayout(lo)

class HsdExpert(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(HsdExpert, self).__init__()

        prefix = pvbase+':'

        lo = QtWidgets.QVBoxLayout()
        
        lo.addWidget(PvPushButton( pvbase+':RESET', 'Reset'))
        lo.addWidget(PvCheckBox( pvbase+':PGPLOOPBACK', 'Loopback'))

        pvtable = [('Even PhaseLock Low'    ,'SYNCELO'),
                   ('Even PhaseLock High'   ,'SYNCEHI'),
                   ('Odd PhaseLock Low'     ,'SYNCOLO'),
                   ('Odd PhaseLock High'    ,'SYNCOHI'),
                   ('Full Threshold(Events)','FULLEVT'),
                   ('Full Size     (Rows)  ','FULLSIZE'),
                   ('Trigger Shift (0..3)  ','TRIGSHIFT'),
                   ('PGP Skp Interval'      ,'PGPSKPINTVL')]
        for elem in pvtable:
            hlo = QtWidgets.QHBoxLayout()
            hlo.addWidget(QtWidgets.QLabel(elem[0]))
            hlo.addWidget(PvEditInt(pvbase+':'+elem[1],''))
            hlo.addStretch(1)
            lo.addLayout(hlo)

        PvLabel( lo, prefix, 'LOCAL12V'  , scale=1 )
        PvLabel( lo, prefix, 'EDGE12V'   , scale=1 )
        PvLabel( lo, prefix, 'AUX12V'    , scale=1 )
        PvLabel( lo, prefix, 'FMC12V'    , scale=1 )
        PvLabel( lo, prefix, 'BOARDTEMP' , scale=1 )
        PvLabel( lo, prefix, 'LOCAL3_3V' , scale=1 )
        PvLabel( lo, prefix, 'LOCAL2_5V' , scale=1 )
        PvLabel( lo, prefix, 'LOCAL1_8V' , scale=1 )
        PvLabel( lo, prefix, 'TOTALPOWER', scale=1 )
        PvLabel( lo, prefix, 'FMCPOWER'  , scale=1 )

        lo.addStretch(1)

        self.setLayout(lo)

class DetBase(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(DetBase, self).__init__()

        prefix = pvbase+':BASE:'
        lo = QtWidgets.QVBoxLayout()

        v = Pv(prefix+'INTTRIGRANGE').get()
        lo.addWidget(QtWidgets.QLabel('Internal Trigger Range: %u - %u clks'%(v[0],v[1])))

        v = Pv(prefix+'INTTRIGCLK').get()
        lo.addWidget(QtWidgets.QLabel('Internal Trigger Clock: %f MHz (%f ns)'%(v,1.e3/v)))

        v = Pv(prefix+'INTTRIGVAL').get()
        lo.addWidget(QtWidgets.QLabel('Internal Trigger Value: %u clks'%v))

        if True:
            hlo = QtWidgets.QHBoxLayout()
            hlo.addWidget(QtWidgets.QLabel('Absolute Trigger Target (usecs)'))
            hlo.addWidget(PvEditInt(prefix+'ABSTRIGTARGET',''))
            hlo.addStretch(1)
            lo.addLayout(hlo)

        v = Pv(prefix+'INTPIPEDEPTH').get()
        lo.addWidget(QtWidgets.QLabel('Internal Pipeline Depth: %u'%v))

        v = Pv(prefix+'INTAFULLVAL').get()
        lo.addWidget(QtWidgets.QLabel('Internal Almost Full Threshold: %u'%v))

        v = Pv(prefix+'MINL0INTERVAL').get()
        lo.addWidget(QtWidgets.QLabel('Minimum Trigger Spacing (usecs): %f'%v))

        v = Pv(prefix+'UPSTREAMRTT').get()
        lo.addWidget(QtWidgets.QLabel('Upstream round trip time (usecs): %f'%v))

        v = Pv(prefix+'DNSTREAMRTT').get()
        lo.addWidget(QtWidgets.QLabel('Downstream round trip time (usecs)): %f'%v))

        if True:
            hlo = QtWidgets.QHBoxLayout()
            hlo.addWidget(QtWidgets.QLabel('PARTITION'))
            hlo.addWidget(PvEditInt(prefix+'PARTITION',''))
            hlo.addStretch(1)
            lo.addLayout(hlo)

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

def main():
    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("base", help="pv base to monitor", default="DAQ:LAB2:HSD")
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
