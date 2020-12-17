import sys
import socket
import argparse
from PyQt5 import QtCore, QtGui, QtWidgets
from psdaq.cas.pvedit import *
from .xpm_utils import *

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
NGroups     = 16
Masks       = ['None','0','1','2','3','4','5','6','7','All']

frLMH       = { 'L':0, 'H':1, 'M':2, 'm':3 }
toLMH       = { 0:'L', 1:'H', 2:'M', 3:'m' }


class PvPAddr(QtWidgets.QWidget):
    def __init__(self, parent, pvbase, name):
        super(PvPAddr,self).__init__()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        label  = QtWidgets.QLabel(name)
        label.setMinimumWidth(100)
        layout.addWidget(label)

        self.__display = PvDisplay()
        self.__display.connect_signal()
        layout.addWidget(self.__display)
        self.setLayout(layout)
        parent.addWidget(self)

        pvname = pvbase+name
        initPvMon(self,pvname)

    def update(self, err):
        q = self.pv.__value__
        if err is None:
            s = '-'
            qs = '%x'%q
            if qs[0:8]=='ffffffff':
                s = 'XTPG'
            elif qs[0:2]=='ff':
                shelf = int(qs[2:3],16)
                port  = int(qs[6:8],16)
                s = 'XPM:%d:AMC%d-%d'%(shelf,port/7,port%7)
            self.__display.valueSet.emit(s)
        else:
            print(err)

class PvPushButtonX(QtWidgets.QPushButton):

    valueSet = QtCore.pyqtSignal('QString',name='valueSet')

    def __init__(self, pvname, label):
        super(PvPushButtonX, self).__init__(label)
        self.setMaximumWidth(70)

        self.clicked.connect(self.buttonClicked)

        self.pv = Pv(pvname, self.update)

    def update(self, err):
        pass

    def buttonClicked(self):
        self.pv.put(1)
        self.pv.put(0)

class PvEditIntX(PvEditInt):

    def __init__(self, pv, label):
        super(PvEditIntX, self).__init__(pv, label)
        self.setMaximumWidth(70)

class PvCmb(PvEditCmb):

    def __init__(self, pvname, choices):
        super(PvCmb, self).__init__(pvname, choices)
        self.setEnabled(False)

class PvGroupMask(PvComboDisplay):
    
    def __init__(self, pvname, label):
        super(PvGroupMask, self).__init__(Masks)
        self.connect_signal()
        self.currentIndexChanged.connect(self.setValue)
        initPvMon(self,pvname)

    def setValue(self):
        ivalue = self.currentIndex()
        value = 0
        if ivalue>8:
            value = 0xff
        elif ivalue>0:
            value = 1<<(ivalue-1)
        if self.pv.__value__ != value:
            self.pv.put(value)
            
    def update(self,err):
        q = self.pv.__value__
        if err is None:
            idx = 0
            if (q&(q-1))!=0:
                idx = 9
            else:
                for i in range(8):
                    if q&(1<<i):
                        idx = i+1
            self.setCurrentIndex(idx)
        else:
            print(err)
                
def LblPushButtonX(parent, pvbase, name, count=1, start=0, istart=0):
    return PvInput(PvPushButtonX, parent, pvbase, name, count, start, istart)

def LblEditIntX(parent, pvbase, name, count=1, start=0, istart=0, enable=True):
    return PvInput(PvEditIntX, parent, pvbase, name, count, start, istart, enable)

def LblGroupMask(parent, pvbase, name, count=1, start=0, istart=0, enable=True):
    return PvInput(PvGroupMask, parent, pvbase, name, count, start, istart, enable)

class PvLinkId:

    def __init__(self,pvname):
        self.linkType = QtWidgets.QLabel('-')
        self.linkType.setMaximumWidth(70)

        self.linkSrc  = QtWidgets.QLabel('-')
        self.linkSrc.setMaximumWidth(70)

        initPvMon(self,pvname)

    def update(self, err):
        value = self.pv.__value__
        names = xpmLinkId(int(value))
        self.linkType.setText(names[0])
        self.linkSrc .setText(names[1])

class PvLinkIdV(QtWidgets.QWidget):

    def __init__(self,pvname,idx):
        super(PvLinkIdV, self).__init__()
        self.pvlink = PvLinkId(pvname)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.pvlink.linkType)
        layout.addWidget(self.pvlink.linkSrc)
        layout.setContentsMargins(0,11,0,11)
        self.setLayout(layout)

class PvLinkIdG:

    def __init__(self,pvname,layout,row,col):
        super(PvLinkIdG, self).__init__()
        self.pvlink = PvLinkId(pvname)
        layout.addWidget(self.pvlink.linkType,row,col)
        layout.addWidget(self.pvlink.linkSrc ,row,col+1)

def FrontPanelAMC(pvbase,iamc):
        dsbox = QtWidgets.QWidget()
        dslo = QtWidgets.QVBoxLayout()
        PvInput(PvLinkIdV, dslo, pvbase, "RemoteLinkId", NDsLinks, start=iamc*NDsLinks)
        LblPushButtonX(dslo, pvbase, "TxLinkReset",    NDsLinks, start=iamc*NDsLinks)
        LblPushButtonX(dslo, pvbase, "RxLinkReset",    NDsLinks, start=iamc*NDsLinks)
        LblPushButtonX(dslo, pvbase, "RxLinkDump" ,    NDsLinks, start=iamc*NDsLinks)
        LblGroupMask  (dslo, pvbase, "LinkGroupMask",  NDsLinks, start=iamc*NDsLinks, enable=True)
        LblCheckBox   (dslo, pvbase, "LinkRxResetDone", NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkRxReady",    NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkTxResetDone", NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkTxReady",    NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkIsXpm",      NDsLinks, start=iamc*NDsLinks, enable=False)
        LblCheckBox   (dslo, pvbase, "LinkLoopback",   NDsLinks, start=iamc*NDsLinks)
#        LblCheckBox  (dslo, pvbase, "LinkRxErr",      NAmcs * NDsLinks, enable=False)
        LblEditIntX   (dslo, pvbase, "LinkRxErr",      NDsLinks, start=iamc*NDsLinks, enable=False)
        LblEditIntX   (dslo, pvbase, "LinkRxRcv",      NDsLinks, start=iamc*NDsLinks, enable=False)
        dslo.addStretch()
        dsbox.setLayout(dslo)
        return dsbox

def DeadTime(pvbase,parent):

    deadbox = QtWidgets.QWidget()
    deadlo = QtWidgets.QVBoxLayout()
    deadgrid = QtWidgets.QGridLayout()

    textWidgets = []
    for j in range(8):
        ptextWidgets = []
        for i in range(32):
            ptextWidgets.append( PvDblArrayW() )
        textWidgets.append(ptextWidgets)

    parent.dtPvId = []
    deadgrid.addWidget( QtWidgets.QLabel('Group'), 0, 0, 1, 2 )
#    deadgrid.addWidget( QtWidgets.QLabel('En'), 0, 2 )
    for j in range(8):
        deadgrid.addWidget( QtWidgets.QLabel('%d'%j ), 0, j+3 )
    for i in range(14):
        parent.dtPvId.append( PvLinkIdG(pvbase+'RemoteLinkId'+'%d'%i,
                                        deadgrid, i+1, 0) )
#        deadgrid.addWidget( PvCheckBox(pvbase+'LinkEnable'+'%d'%i,None), i+1, 2 )
        for j in range(8):
            deadgrid.addWidget( textWidgets[j][i], i+1, j+3 )
    for i in range(16,21):
        k = i-1
        deadgrid.addWidget( QtWidgets.QLabel('BP-slot%d'%(i-13)), k, 0, 1, 2 )
#        deadgrid.addWidget( PvCheckBox(pvbase+'LinkEnable'+'%d'%(i+1),None), k, 2 )
        for j in range(8):
            deadgrid.addWidget( textWidgets[j][i], k, j+3 )
    for i in range(28,32):
        k = i-7
        deadgrid.addWidget( QtWidgets.QLabel('INH%d'%(i-28)), k, 0, 1, 2 )
#        deadgrid.addWidget( PvCheckBox(pvbase+'LinkEnable'+'%d'%i,None), k, 2 )
        for j in range(8):
            deadgrid.addWidget( textWidgets[j][i], k, j+3 )

    parent.deadflnk = []
    for j in range(8):
        ppvbase = pvbase+'PART:%d:'%j
        print(ppvbase)
        parent.deadflnk.append( PvDblArray( ppvbase+'DeadFLnk', textWidgets[j] ) )

    deadlo.addLayout(deadgrid)
    deadlo.addStretch()
    deadbox.setLayout(deadlo)
    return deadbox

class PvRxAlign(QtWidgets.QWidget):
    def __init__(self,pvname,title):
        super(PvRxAlign, self).__init__()
        layout = QtWidgets.QHBoxLayout()

        pv = Pv(pvname)
        v = pv.get()

        label = title+'\n'+str(v[0])
        layout.addWidget( QtWidgets.QLabel(label) )

        self.image   = QtGui.QImage(64,20,QtGui.QImage.Format_Mono)
        painter = QtGui.QPainter(self.image)
        painter.fillRect(0,0,64,20,QtGui.QColor(255,255,255))

        painter.setBrush(QtGui.QColor(0,0,255))   # blue
        max = 1
        for i in range(64):
            if v[i+1] > max:
                max = v[i+1]

        for i in range(64):
            q = 16*v[i+1]/max
            painter.drawLine(i,15-q,i,15)

        painter.setBrush(QtGui.QColor(255,0,0))   # red
        i = v[0]+1
        painter.drawLine(i,16,i,20)

        canvas = QtWidgets.QLabel()
        canvas.setPixmap(QtGui.QPixmap.fromImage(self.image))

        layout.addWidget( canvas )
        self.setLayout(layout)

def addTiming(self,pvbase):
    lor = QtWidgets.QVBoxLayout()
    PvLabel(self,lor, pvbase, "RxClks"     )
    PvLabel(self,lor, pvbase, "TxClks"     )
    PvLabel(self,lor, pvbase, "RxRsts"     )
    PvLabel(self,lor, pvbase, "CrcErrs"    )
    PvLabel(self,lor, pvbase, "RxDecErrs"  )
    PvLabel(self,lor, pvbase, "RxDspErrs"  )
    PvLabel(self,lor, pvbase, "BypassRsts" )
    PvLabel(self,lor, pvbase, "BypassDones")
    PvLabel(self,lor, pvbase, "RxLinkUp"   )
    PvLabel(self,lor, pvbase, "FIDs"       )
    PvLabel(self,lor, pvbase, "SOFs"       )
    PvLabel(self,lor, pvbase, "EOFs"       )
    LblPushButtonX( lor, pvbase, "RxReset" )
    lor.addWidget( PvRxAlign(pvbase+'RxAlign','RxAlign') )
    lor.addStretch()
    w = QtWidgets.QWidget()
    w.setLayout(lor)
    return w

class PvMmcm(QtWidgets.QWidget):
    def __init__(self,pvname,rstname,title):
        super(PvMmcm, self).__init__()
        layout = QtWidgets.QHBoxLayout()

        pv = Pv(pvname)
#        v = pv.get()
        v = pv.get()

        v0 = int(v[0])
        iedge   = v0 & 0xffff
        inum    = (v0>>16)&0x1fff
        ibusy   = (v0>>30)&1
        ilocked = (v0>>31)&1
        
        label = title+'\n'+str(iedge)
        if ilocked==1:
            label += '*'
        if ibusy==1:
            label += 'R'
        layout.addWidget( QtWidgets.QLabel(label) )

        layout.addWidget( PvPushButton(rstname,"R") )

        w = int(inum/4)
        self.image   = QtGui.QImage(w,20,QtGui.QImage.Format_Mono)
        painter = QtGui.QPainter(self.image)
        painter.fillRect(0,0,w,20,QtGui.QColor(255,255,255))

        painter.setBrush(QtGui.QColor(0,0,255))   # blue
        for i in range(w):
            q = 0
            for j in range(4):
                q += int(v[i*4+j+1])
            painter.drawLine(i,15-(q>>4),i,15)

        painter.setBrush(QtGui.QColor(255,0,0))   # red
        i = int(iedge/4)
        painter.drawLine(i,16,i,20)

        canvas = QtWidgets.QLabel()
        canvas.setPixmap(QtGui.QPixmap.fromImage(self.image))

        layout.addWidget( canvas )
        self.setLayout(layout)

def intInput(layout, pv, label):
    lo = QtWidgets.QHBoxLayout()
    lo.addWidget( QtWidgets.QLabel(label) )
    lo.addWidget( PvEditInt(pv, '') )
    layout.addLayout(lo)

def addCuTab(self,pvbase):
    lor = QtWidgets.QVBoxLayout()
    lor.addWidget( addTiming(self,pvbase+'Cu:') )

    LblEditIntX( lor, pvbase+'XTPG:', 'CuInput'   )
    LblEditIntX( lor, pvbase+'XTPG:', 'CuBeamCode')
    LblEditIntX( lor, pvbase+'XTPG:', 'CuDelay'   )
    PvLabel(self,lor, pvbase+'XTPG:', 'CuDelay_ns')
    PvLabel(self,lor, pvbase+'XTPG:', 'PulseId', isInt=True)
    PvLabel(self,lor, pvbase+'XTPG:', 'TimeStamp', isTime=True)
    PvLabel(self,lor, pvbase+'XTPG:', 'FiducialIntv', isInt=True)
    LblCheckBox( lor, pvbase+'XTPG:', 'FiducialErr', enable=False)
    LblPushButtonX( lor, pvbase+'XTPG:', 'ClearErr' )

    for i in range(4):
        lor.addWidget( PvMmcm(pvbase+'XTPG:MMCM%d'%i , pvbase+'XTPG:ResetMmcm%d'%i, 'mmcm%d'%i) )

    lor.addStretch()
    w = QtWidgets.QWidget()
    w.setLayout(lor)
    return w
    
class Ui_MainWindow(object):
    def setupUi(self, MainWindow, titles):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self._pvlabels = []

        tabsel = QtWidgets.QComboBox()
        tabsel.addItems(titles)

        stack = QtWidgets.QStackedWidget()

        for title in titles:
            pvbase = title + ':'

            tw  = QtWidgets.QTabWidget()

            tb  = QtWidgets.QWidget()
            hl  = QtWidgets.QVBoxLayout()
            #        PvLabel  (hl, pvbase, "PARTITIONS"  )
            #            PvLabel  (hl, pvbase, "PAddr"       , isInt=True)
            PvPAddr  (hl, pvbase, "PAddr"       )
            PvCString(hl, pvbase, "FwBuild"     )

            LblPushButtonX(hl, pvbase, "ModuleInit"      )
            LblPushButtonX(hl, pvbase, "DumpPll",        NAmcs)
            LblPushButtonX(hl, pvbase, "DumpTiming",     2)
            LblPushButtonX(hl, pvbase, "DumpSeq"         )
#            LblEditIntX   (hl, pvbase, "SetVerbose"      )
            LblPushButtonX(hl, pvbase, "Inhibit"         )
            LblPushButtonX(hl, pvbase, "TagStream"       )
            PvLabel(self, hl, pvbase, "RecClk"     )
            PvLabel(self, hl, pvbase, "FbClk"      )
            PvLabel(self, hl, pvbase, "BpClk"      )
            hl.addStretch()
            tb.setLayout(hl)
            tw.addTab(tb,"Global")

            pv = Pv(pvbase+'FwBuild')
            v = pv.get()
            if 'xtpg' in v:
                tw.addTab( addCuTab (self,pvbase      ), "CuTiming")
            else:
                tw.addTab( addTiming(self,pvbase+'Us:'), "UsTiming")

            tw.addTab(FrontPanelAMC(pvbase,0),"AMC0")
            tw.addTab(FrontPanelAMC(pvbase,1),"AMC1")

            bpbox  = QtWidgets.QWidget()
            bplo   = QtWidgets.QVBoxLayout()
            LblPushButtonX(bplo, pvbase, "TxLinkReset16",    1, 16, 0)
            LblCheckBox   (bplo, pvbase, "LinkTxReady16",    1, 16, 0, enable=False)

            #LblPushButtonX(bplo, pvbase, "TxLinkReset",    5, 17, 3)
            LblPushButtonX(bplo, pvbase, "RxLinkReset",    5, 17, 3)
            #LblCheckBox   (bplo, pvbase, "LinkEnable",     5, 17, 3)
            LblCheckBox   (bplo, pvbase, "LinkRxReady",    5, 17, 3, enable=False)
            #LblCheckBox  (bplo, pvbase, "LinkTxReady",    5, 17, 3, enable=False)
            #LblCheckBox  (bplo, pvbase, "LinkIsXpm",      5, 17, 3, enable=False)
            #LblCheckBox  (bplo, pvbase, "LinkLoopback",   5, 17, 3)
            LblEditIntX   (bplo, pvbase, "LinkRxErr",      5, 17, 3, enable=False)
            LblEditIntX   (bplo, pvbase, "LinkRxRcv",      5, 17, 3, enable=False)
            bplo.addStretch()
            bpbox.setLayout(bplo)
            tw.addTab(bpbox,"Bp")

            pllbox  = QtWidgets.QWidget()
            pllvbox = QtWidgets.QVBoxLayout() 
            LblCheckBox  (pllvbox, pvbase, "PLL_LOS",        NAmcs, enable=False)
            LblEditIntX  (pllvbox, pvbase, "PLL_LOSCNT",     NAmcs, enable=False)
            LblCheckBox  (pllvbox, pvbase, "PLL_LOL",        NAmcs, enable=False)
            LblEditIntX  (pllvbox, pvbase, "PLL_LOLCNT",     NAmcs, enable=False)
            pllvbox.addStretch()
            pllbox.setLayout(pllvbox)
            tw.addTab(pllbox,"PLLs")

            tw.addTab(DeadTime(pvbase,self),"DeadTime")

            stack.addWidget(tw)

        lol = QtWidgets.QVBoxLayout()
        lol.addWidget(tabsel)
        lol.addWidget(stack)

        ltable = QtWidgets.QWidget()
        ltable.setLayout(lol)

        lscroll = QtWidgets.QScrollArea()
        lscroll.setWidget(ltable)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(lscroll)

        self.centralWidget.setLayout(layout)
        self.centralWidget.resize(720,600)

        tabsel.currentIndexChanged.connect(stack.setCurrentIndex)

        MainWindow.resize(720,600)
        MainWindow.setWindowTitle('xpmpva')
        MainWindow.setCentralWidget(self.centralWidget)

def main():
    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("pvs", help="pvs to monitor",nargs='+')
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.pvs)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
