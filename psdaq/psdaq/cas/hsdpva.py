import sys
import argparse
import logging
import socket
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from psdaq.cas.pvedit import *
from p4p.client.thread import Context
from psdaq.hsd.pvdef import *

NChannels = 1
NLanes = 4
NBuffers = 16
Patterns = ['0: Data','4: Ramp','5: Transport']

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

class PvScalarBox(QtWidgets.QGroupBox):
    def __init__(self, pvname, title, struct, edit=False):
        super(PvScalarBox,self).__init__(title)
        self.struct = struct
        glo = QtWidgets.QGridLayout()
        self.widgets = []
        for i,ttl in enumerate(struct):
            glo.addWidget(QtWidgets.QLabel(ttl),i,0)
            if edit:
                w = QtWidgets.QLineEdit('-')
            else:
                w = QtWidgets.QLabel('-')
            glo.addWidget(w,i,1)
            if len(struct[ttl])>2:
                glo.addWidget(QtWidgets.QLabel(struct[ttl][2]),i,2)
            self.widgets.append(w)

        if edit:
            b = QtWidgets.QPushButton('Apply')
            glo.addWidget(b,len(struct),1)
            glo.setRowStretch(len(struct)+1,1)
            b.clicked.connect(self.put)
        else:
            glo.setRowStretch(len(struct),1)
        glo.setColumnStretch(2,1)
        self.setLayout(glo)
        initPvMon(self,pvname,isStruct=True)

    def update(self,err):
        if err is None:
            q = self.pv.__value__.todict()
            for i,v in enumerate(q):
                self.widgets[i].setText(QString(q[v]))

    def put(self):
        v = {}
        w = self.widgets
        for i,ttl in enumerate(self.struct):
            val = int(w[i].text()) if self.struct[ttl][0]=='i' else float(w[i].text())
            v[ttl] = val
        print(v)
        self.pv.put(v)

class PvBuf(PvScalarBox):
    def __init__( self, pvname, title):
        super(PvBuf,self).__init__(pvname, title, monBuf)

class PvArrayTable(QtWidgets.QGroupBox):
    def __init__( self, pvname, title, struct, vertical=False, edit=False):
        super(PvArrayTable,self).__init__(title)
        self.struct = struct
        glo = QtWidgets.QGridLayout()
        self.widgets = []
        for i,ttl in enumerate(struct):
            x = 0 if vertical else i
            y = i if vertical else 0
            glo.addWidget(QtWidgets.QLabel(ttl),x,y)
            wl = []
            for j in range(len(struct[ttl][1])):
                x = j+1 if vertical else i
                y = i   if vertical else j+1
                w = QtWidgets.QLineEdit('-') if edit else QtWidgets.QLabel('')
                glo.addWidget(w,x,y)
                wl.append(w)
            self.widgets.append(wl)

        if edit:
            b = QtWidgets.QPushButton('Apply')
            glo.addWidget(b,x+1,1)
            x += 1
            b.clicked.connect(self.put)

        glo.setRowStretch   (x+1,1)
        glo.setColumnStretch(y+1,1)
        self.setLayout(glo)

        initPvMon(self,pvname,isStruct=True)

    def update(self,err):
        if err is None:
            q = self.pv.__value__.todict()
            for i,v in enumerate(q):
                for j,w in enumerate(q[v]):
                    if len(self.struct[v])>2:
                        self.widgets[i][j].setText(QString(self.struct[v][2].format(w)))
                    else:
                        self.widgets[i][j].setText(QString(w))

    def put(self):
        d = {}
        for i,v in enumerate(self.struct):
            val = []
            for j,w in enumerate(self.struct[v][1]):
                #if len(self.struct[v])>2:
                #    val.append(int(w[i][j].text()))
                val.append(int(self.widgets[i][j].text()))
                    # or float
            d[v] = tuple(val)

        self.pv.put(d)

class PvJesd(object):
    def __init__( self, pvname, statWidgets, clockWidgets):
        self.statWidgets =statWidgets
        self.clockWidgets=clockWidgets
        initPvMon(self,pvname,isStruct=True)

    def update(self,err):
        if err is None:
            q = self.pv.__value__.todict()
            for i,v in enumerate(q['stat']):
                self.statWidgets[i].setText(QString(v))
            for i,v in enumerate(q['clks']):
                self.clockWidgets[i].setText(QString('{0:.4f}'.format(v)))

class PvPLink(QtWidgets.QWidget):
    def __init__( self, parent, pvbase, name ):
        super(PvPLink,self).__init__()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        label  = QtWidgets.QLabel(name)
        label.setMinimumWidth(100)
        layout.addWidget(label)
        self.__display = PvDisplay()
        self.__display.setWordWrap(True)
        self.__display.connect_signal()
        layout.addWidget(self.__display)
        self.setLayout(layout)
        parent.addWidget(self)

        pvname = pvbase+name
        initPvMon(self,pvname,True)

    def update(self,err):
        q = self.pv.get().value
        # transform q into s
        s = 'Unknown'
        itype = (int(q)>>24)&0xff
        if (itype == 0xfb) and (q&0xffff)!=0:
            ip_addr = '172.21'+'.%u'%((int(q)>>8)&0xff)+'.%u'%((int(q)>>0)&0xff)
            host = socket.gethostbyaddr(ip_addr)[0].split('.')[0].split('-')[-1]
            s = host+':{:}'.format((q>>16)&0xff)
        if err is None:
            self.__display.valueSet.emit(s)
        else:
            print(err)

class HsdConfig(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(HsdConfig, self).__init__()
        self._rows = []
        self._pvlabels = []

        lo = QtWidgets.QVBoxLayout()
        glo = QtWidgets.QGridLayout()

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
        self.widgets = {}
        i=0
        for pv in daqConfig:
            self.widgets[pv] = QtWidgets.QLineEdit('-')
            glo.addWidget(QtWidgets.QLabel(pv), i, 0)
            i=i+1
        lo.addLayout(glo)

        if True:
            hlo = QtWidgets.QHBoxLayout()
            hlo.addWidget(QtWidgets.QLabel('Test Pattern'))
            hlo.addWidget(PvEditInt(pvbase+':TESTPATTERN',''))
            hlo.addWidget(QtWidgets.QLabel('\n'.join(Patterns)))
            hlo.addStretch(1)
            lo.addLayout(hlo)

        lo.addWidget(PvPushButton ( pvbase+':BASE:APPLYCONFIG'  , 'Enable'))
        lo.addWidget(PvPushButton ( pvbase+':BASE:APPLYUNCONFIG', 'Disable'))
        PvLabel      (self, lo, pvbase+':BASE:', 'READY'      , isInt=False)

        lo.addStretch(1)
        
        self.setLayout(lo)

class HsdBufferSummary(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(HsdBufferSummary,self).__init__()

        lo = QtWidgets.QVBoxLayout()
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(PvBuf(pvbase+':MONRAWBUF','Raw'))
        hb.addWidget(PvBuf(pvbase+':MONFEXBUF','Fex'))
        lo.addLayout(hb)

        if False:
            glo = QtWidgets.QGridLayout()
            self._rows.append(PvRow( glo,0, pvbase+':DATA_FIFOOF'    , 'Data Fifo Overflow' , False, MaxLen=NChannels))
            self._rows.append(PvRow( glo,1, pvbase+':WRFIFOCNT' , 'Write FIFO Count', False, MaxLen=NChannels))
            self._rows.append(PvRow( glo,2, pvbase+':RDFIFOCNT' , 'Read FIFO Count', False, MaxLen=NChannels))
            lo.addLayout(glo)

        lo.addStretch(1)
        self.setLayout(lo)

class HsdEnv(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(HsdEnv, self).__init__()
        lo = QtWidgets.QVBoxLayout()
        PvCString(lo, pvbase+':', 'FWBUILD',isStruct=True)
        PvCString(lo, pvbase+':', 'PADDR',isStruct=True)
        PvPLink  (lo, pvbase+':', 'PLINK')
#        buildpv = Pv(pvbase+':FWBUILD')
#        lo.addWidget( QtWidgets.QLabel(buildpv.get().replace(',','\n')) )
#        paddrpv = Pv(pvbase+':PADDR')
#        lo.addWidget( QtWidgets.QLabel(paddrpv.get()) )
        lo.addWidget( PvScalarBox(pvbase+':MONENV','Env',monEnv) )
        lo.addStretch()
        self.setLayout(lo)

class HsdJesd(QtWidgets.QWidget):

    def __init__(self, pvbase):
        super(HsdJesd, self).__init__()
        self._pvlabels = []

        pvTtl = Pv(pvbase+':MONJESDTTL',isStruct=True)
        pvTtl.get()
        
        lo = QtWidgets.QGridLayout()

        swidgets = []
        lo.addWidget(QtWidgets.QLabel('Lane'),0,0)
        for i in range(8):
            lo.addWidget(QtWidgets.QLabel('%d'%i),0,i+1)
        for i,ttl in enumerate(pvTtl.__value__.ttl):
            lo.addWidget(QtWidgets.QLabel(ttl),i+1,0)
        for j in range(8):
            for i in range(len(pvTtl.__value__.ttl)):
                w = QtWidgets.QLabel('')
                lo.addWidget(w,i+1,j+1)
                swidgets.append(w)

        vlo = QtWidgets.QVBoxLayout()
        vlo.addLayout(lo)
        vlo.addStretch()

        glo = QtWidgets.QGridLayout()
        cwidgets = []
        names = ['PllClk','RxClk','SysRef','DevClk','GtRefClk']
        for i in range(len(names)):
            glo.addWidget(QtWidgets.QLabel(names[i]),i,0)
            w = QtWidgets.QLabel('')
            glo.addWidget(w,i,1)
            cwidgets.append(w)
            glo.addWidget(QtWidgets.QLabel('MHz'),i,2)
        glo.setColumnStretch(3,1)
        vlo.addLayout(glo)
        vlo.addStretch()

        self.monjesd = PvJesd(pvbase+':MONJESD',swidgets,cwidgets)

        self.setLayout(vlo)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, titles):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        tabsel = QtWidgets.QComboBox()
        tabsel.addItems(titles)

        stack = QtWidgets.QStackedWidget()
        
        for title in titles:
            maintab = QtWidgets.QTabWidget()
            maintab.addTab( PvScalarBox(title+':CONFIG','Config',daqConfig,edit=True),
                            'Config' )
            maintab.addTab( PvScalarBox(title+':MONTIMING','Timing',monTiming),
                            'Timing' )
            maintab.addTab( PvArrayTable(title+':MONPGP','Pgp',monPgp),
                            'PGP' )
            maintab.addTab( PvArrayTable(title+':PGPCONFIG','Pgp Config',pgpConfig,edit=True),
                            'PgpCfg' )
            maintab.addTab( HsdBufferSummary(title), 
                            'Buffers' )
            maintab.addTab( PvArrayTable(title+':MONRAWDET','Raw Buffers',monBufDetail,vertical=True), 
                            'RawDetail' )
            maintab.addTab( PvArrayTable(title+':MONFEXDET','Fex Buffers',monBufDetail,vertical=True), 
                            'FexDetail' )
            maintab.addTab( PvScalarBox(title+':MONFLOW','Flow',monFlow),
                            'Flow' )
            maintab.addTab( HsdEnv          (title),
                            'Env' )
            maintab.addTab( PvScalarBox(title+':MONADC','Adc',monAdc),
                            'Adc' )
            maintab.addTab( HsdJesd         (title), 
                            'Jesd' )
            maintab.addTab( PvScalarBox(title+':RESET','Reset',daqReset,edit=True),
                            'Reset' )
            stack.addWidget(maintab)

        lo = QtWidgets.QVBoxLayout()
        lo.addWidget(tabsel)
        lo.addWidget(stack)

        self.centralWidget.setLayout(lo)

        tabsel.currentIndexChanged.connect(stack.setCurrentIndex)
        
        MainWindow.resize(500,550)
        MainWindow.setWindowTitle("hsdpva")
        MainWindow.setCentralWidget(self.centralWidget)

def main():
    global NChannels
    global NLanes
    global Patterns

    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("base", help="pv base to monitor", nargs='+', default="DAQ:LAB2:HSD:DEV06_3E:A")
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
