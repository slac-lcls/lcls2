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

NAmcs       = 2
NGroups     = 8
NSeqCodes   = 32
Masks       = ['None','0','1','2','3','4','5','6','7','All']

frLMH       = { 'L':0, 'H':1, 'M':2, 'm':3 }
toLMH       = { 0:'L', 1:'H', 2:'M', 3:'m' }

ATCAWidget  = None

def isATCA(v):
    result = ('Kcu1500' not in v) and ('C1100' not in v)
    print(f'isATCA({v}) = {result}')
    return result

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
                shelf = int(qs[2:4],16)
                port  = int(qs[6:8],16)

                pvbase = ':'.join(self.pv.pvname.split(':')[:-2])+f':{shelf}'
                pv = Pv(pvbase+':FwBuild')
                v = pv.get()
                if isATCA(v):
                    s = 'XPM:%d:AMC%d-%d'%(shelf,port/7,port%7)
                else:
                    s = 'XPM:%d:QSFP%d-%d'%(shelf,port/4,port%4)
                    
            self.__display.valueSet.emit(s)
        else:
            print(err)

class PvPushButtonX(QtWidgets.QPushButton):

    valueSet = QtCore.pyqtSignal('QString',name='valueSet')

    def __init__(self, pvname, label):
        super(PvPushButtonX, self).__init__(label)
        if ATCAWidget:
            self.setMaximumWidth(70)

        self.clicked.connect(self.buttonClicked)

        self.pv = Pv(pvname, self.update)

    def update(self, err):
        pass

    def buttonClicked(self):
        self.pv.put(1)
        self.pv.put(0)

class PvPushButtonVal(QtWidgets.QPushButton):

    valueSet = QtCore.pyqtSignal('QString',name='valueSet')

    def __init__(self, pvname, label, value):
        super(PvPushButtonVal, self).__init__(label)
        self.value = value
        if ATCAWidget:
            self.setMaximumWidth(70)

        self.clicked.connect(self.buttonClicked)

        initPvMon(self,pvname)

    def update(self, err):
        pass

    def buttonClicked(self):
        self.pv.put(self.value)

class PvEditIntX(PvEditInt):

    def __init__(self, pv, label):
        super(PvEditIntX, self).__init__(pv, label)
        if ATCAWidget:
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
        if ivalue>NGroups:
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
                idx = NGroups+1
            else:
                for i in range(NGroups):
                    if q&(1<<i):
                        idx = i+1
            self.setCurrentIndex(idx)
        else:
            print(err)

def LblPushButtonX(parent, pvbase, name, count=1, start=0, istart=0, label=None):
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

def FrontPanelAMC(pvbase,nDsLinks,start):
        dsbox = QtWidgets.QWidget()
        dslo = QtWidgets.QGridLayout()
        headers = [("RemoteLinkId"   ,PvLinkIdV    ,True),
                   ("TxLinkReset"    ,PvPushButtonX,True),
                   ("RxLinkReset"    ,PvPushButtonX,True),
                   ("RxLinkDump"     ,PvPushButtonX,True),
                   ("LinkGroupMask"  ,PvGroupMask  ,True),
                   ("LinkRxResetDone",PvCheckBox   ,False),
                   ("LinkRxReady"    ,PvCheckBox   ,False),
                   ("LinkTxResetDone",PvCheckBox   ,False),
                   ("LinkTxReady"    ,PvCheckBox   ,False),
                   ("LinkIsXpm"      ,PvCheckBox   ,False),
                   ("LinkLoopback"   ,PvCheckBox   ,True),
                   ("LinkRxErr"      ,PvEditIntX   ,False),
                   ("LinkRxRcv"      ,PvEditIntX   ,False)]
        for row,h in enumerate(headers):
            dslo.addWidget(QtWidgets.QLabel(h[0]), row, 0 )
            for col in range(nDsLinks):
                w = h[1](f'{pvbase}{h[0]}{col+start}',f'{col+start}')
                w.setEnabled(h[2])
                dslo.addWidget(w, row, col+1)
        dslo.setRowStretch(len(headers),1)
        dslo.setColumnStretch(nDsLinks+1,1)
        dsbox.setLayout(dslo)
        return dsbox

def PLLs(pvbase,ncol):
        dsbox = QtWidgets.QWidget()
        dslo = QtWidgets.QGridLayout()
        headers = [("PLL_LOS"   ,PvCheckBox    ,False),
                   ("PLL_LOSCNT",PvEditIntX    ,False),
                   ("PLL_LOL"   ,PvCheckBox    ,False),
                   ("PLL_LOLCNT",PvEditIntX    ,False)]
        for row,h in enumerate(headers):
            dslo.addWidget(QtWidgets.QLabel(h[0]), row, 0 )
            for col in range(ncol):
                w = h[1](f'{pvbase}{h[0]}{col}',f'{col}')
                w.setEnabled(h[2])
                dslo.addWidget(w, row, col+1)
        dslo.setRowStretch(len(headers),1)
        dslo.setColumnStretch(ncol+1,1)
        dsbox.setLayout(dslo)
        return dsbox

def DeadTime(pvbase,parent):

    deadbox = QtWidgets.QWidget()
    deadlo = QtWidgets.QVBoxLayout()
    deadgrid = QtWidgets.QGridLayout()

    textWidgets = []
    for j in range(NGroups):
        ptextWidgets = []
        for i in range(32):
            ptextWidgets.append( PvDblArrayW() )
        textWidgets.append(ptextWidgets)

    parent.dtPvId = []
    deadgrid.addWidget( QtWidgets.QLabel('Group'), 0, 0, 1, 2 )
#    deadgrid.addWidget( QtWidgets.QLabel('En'), 0, 2 )
    for j in range(NGroups):
        deadgrid.addWidget( QtWidgets.QLabel('%d'%j ), 0, j+3 )
    for i in range(14):
        parent.dtPvId.append( PvLinkIdG(pvbase+'RemoteLinkId'+'%d'%i,
                                        deadgrid, i+1, 0) )
#        deadgrid.addWidget( PvCheckBox(pvbase+'LinkEnable'+'%d'%i,None), i+1, 2 )
        for j in range(NGroups):
            deadgrid.addWidget( textWidgets[j][i], i+1, j+3 )
    for i in range(28,32):
        k = i-12
        deadgrid.addWidget( QtWidgets.QLabel('INH%d'%(i-28)), k, 0, 1, 2 )
#        deadgrid.addWidget( PvCheckBox(pvbase+'LinkEnable'+'%d'%i,None), k, 2 )
        for j in range(8):
            deadgrid.addWidget( textWidgets[j][i], k, j+3 )

    parent.deadflnk = []
    for j in range(NGroups):
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
            q = int(16*v[i+1]/max)
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
    PvLabel(self,lor, pvbase, "RxRsts"     )
    PvLabel(self,lor, pvbase, "CrcErrs"    )
    PvLabel(self,lor, pvbase, "RxDecErrs"  )
    PvLabel(self,lor, pvbase, "RxDspErrs"  )
    PvLabel(self,lor, pvbase, "BypassRsts" )
    PvLabel(self,lor, pvbase, "BypassDones")
    PvLabel(self,lor, pvbase, "RxLinkUp"   )
    PvLabel(self,lor, pvbase, "FIDs"       )
    LblPushButtonX( lor, pvbase, "RxReset" )
    LblPushButtonX( lor, pvbase, "RxCountReset" )
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

def addUsTab(self,pvbase):
    lor = QtWidgets.QVBoxLayout()
    lor.addWidget( addTiming(self,pvbase+'Us:') )
#   xpmGenKcu1500 can't generate this PV???
#    lor.addWidget( PvMmcm(pvbase+'XTPG:MMCM3', pvbase+'XTPG:ResetMmcm3', 'mmcm3') )

    w = QtWidgets.QWidget()
    w.setLayout(lor)
    return w

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

class XpmGroups(object):
    # monitor PAddr recursively
    # monitor PART:[0..7].Master,L0InpRate
    def __init__(self,pvbase):
    # assuming that pvbase is of the form DAQ:NEH:XPM:1:
        pvbase_split = pvbase.split(":")
        pos = pvbase_split.index("XPM")
        self.name = ":".join([pvbase_split[pos],pvbase_split[pos+1],""])
        self.parent  = None
        # adding try except because in the fee teststand xpm0 is not reachable
        paddr = Pv(pvbase+'PAddr').get()

        if paddr!=0xffffffff:
            name = xpmLinkId(paddr)[0]
            if 'XPM' in name:
                xpmpath=":".join(pvbase_split[:pos])
                xpmname=":".join(["",name,""])
                self.parent = XpmGroups(f'{xpmpath}{xpmname}')

        self.vals = {'master':{i:Pv(pvbase+f'PART:{i}:Master'    ,self.update) for i in range(NGroups)},
                     'l0rate':{i:Pv(pvbase+f'PART:{i}:L0InpRate' ,self.update) for i in range(NGroups)},
                     'codes' : Pv(pvbase+f'SEQCODES'             ,self.update, isStruct=True) }

    def update(self,err):
        pass

    def _update(self):
        if self.parent:
            vals = self.parent._update()
        else:
            vals = {'master':{i:'-' for i in range(NGroups)},
                    'l0rate':{i:'-' for i in range(NGroups)},
                    'codes' :{i:{'master':'-',
                                 'desc'  :'-',
                                 'rate'  :'-'} for i in range(NSeqCodes)}}
        for i in range(NGroups):
            v = self.vals['master'][i].__value__
            if v is not None and v > 0:
                vals['master'][i] = self.name
                vals['l0rate'][i] = str(self.vals['l0rate'][i].__value__)

        codesv = self.vals['codes'].__value__
        if codesv:
            codes = codesv.todict()['value']
            for i in range(NSeqCodes):
                if codes['Enabled'][i]:
                    vals['codes'][i] = {'master':self.name,
                                        'desc'  :codes['Description'][i],
                                        'rate'  :str(codes['Rate'][i])}
        return vals

class GroupsTab(QtWidgets.QWidget):
    def __init__(self, pvbase):
        super(GroupsTab,self).__init__()

        l = QtWidgets.QHBoxLayout()
        lv = QtWidgets.QVBoxLayout()
        grid1 = QtWidgets.QGridLayout()
        grid1.addWidget( QtWidgets.QLabel('Group')    , 0, 0 )
        grid1.addWidget( QtWidgets.QLabel('Master')   , 0, 1 )
        grid1.addWidget( QtWidgets.QLabel('L0InpRate'), 0, 2 )
        self.masterText = {}
        self.l0RateText = {}
        for i in range(NGroups):
            grid1.addWidget( QtWidgets.QLabel(str(i)), i+1, 0 )
            self.masterText[i] = QtWidgets.QLabel('None')
            grid1.addWidget( self.masterText[i], i+1, 1)
            self.l0RateText[i] = QtWidgets.QLabel('-')
            grid1.addWidget( self.l0RateText[i], i+1, 2)
#        grid1.setRowStretch(grid1.rowCount(),1)
        box1 = QtWidgets.QGroupBox("Groups")
        box1.setLayout(grid1)

        grid3 = QtWidgets.QGridLayout()
        grid3.addWidget( QtWidgets.QLabel('Sequence'), 0, 0)
        for i in range(NSeqCodes//4):
            grid3.addWidget( QtWidgets.QLabel(str(i)), 1+i, 0)
            grid3.addWidget( PvPushButtonVal(f'{pvbase}SEQENG:{i}:ENABLE', 'Ena', 1), 1+i, 1 )
            grid3.addWidget( PvPushButtonVal(f'{pvbase}SEQENG:{i}:ENABLE', 'Dis', 0), 1+i, 2 )
            grid3.addWidget( PvPushButtonVal(f'{pvbase}SEQENG:{i}:DUMP', 'Dump', 1), 1+i,3 )
#        grid3.setRowStretch(grid3.rowCount(),1)
        box3 = QtWidgets.QGroupBox("Sequence Control")
        box3.setLayout(grid3)
            
        lv.addWidget(box1)
        lv.addWidget(box3)
        lv.addStretch()
        l.addLayout(lv)
        l.addStretch()

        grid2 = QtWidgets.QGridLayout()
        grid2.addWidget( QtWidgets.QLabel('EventCode')  , 0, 0 )
        grid2.addWidget( QtWidgets.QLabel('Master')     , 0, 1 )
        grid2.addWidget( QtWidgets.QLabel('Description'), 0, 2 )
        grid2.addWidget( QtWidgets.QLabel('Rate'       ), 0, 3 )
        self.codesText = {'master':{},
                          'desc'  :{},
                          'rate'  :{}}
        for i in range(NSeqCodes):
            grid2.addWidget( QtWidgets.QLabel(str(i+288-NSeqCodes)), i+1, 0 )
            self.codesText['master'][i] = QtWidgets.QLabel('None')
            grid2.addWidget( self.codesText['master'][i], i+1, 1 )
            self.codesText['desc'][i] = QtWidgets.QLabel('-')
            grid2.addWidget( self.codesText['desc'][i], i+1, 2 )
            self.codesText['rate'][i] = QtWidgets.QLabel('-')
            grid2.addWidget( self.codesText['rate'][i], i+1, 3 )
        grid2.setRowStretch(grid2.rowCount(),1)
        box2 = QtWidgets.QGroupBox("Event Code Sources")
        box2.setLayout(grid2)
        l.addWidget(box2)

        self.setLayout(l)

        self.xpm = XpmGroups(pvbase)

        initPvMon(self,pvbase+'SEQCODES',isStruct=True)

    def update(self,err):
        vals = self.xpm._update()
        for i in range(NGroups):
            self.masterText[i].setText(vals['master'][i])
            self.l0RateText[i].setText(vals['l0rate'][i])
        for i in range(NSeqCodes):
            self.codesText['master'][i].setText(vals['codes'][i]['master'])
            self.codesText['desc'  ][i].setText(vals['codes'][i]['desc'  ])
            self.codesText['rate'  ][i].setText(vals['codes'][i]['rate'  ])

class PatternTab(QtWidgets.QWidget):
    def __init__(self, pvbase):
        super(PatternTab,self).__init__()

        l = QtWidgets.QVBoxLayout()
        l.addWidget(PvEditEvt(f'{pvbase}PATT:L0Select',0))

        v20b = (1<<20)-1
        l.addWidget(PvTableDisplay(f'{pvbase}PATT:GROUPS', [f'Group{i}' for i in range(8)], (0, v20b, v20b, v20b, 0)))

        self.coinc = []
        g = QtWidgets.QGridLayout()
        for i in range(8):
            g.addWidget(QtWidgets.QLabel(f'G{i}'),0,i+1)
            g.addWidget(QtWidgets.QLabel(f'G{i}'),i+1,0)
        for i in range(8):
            for j in range(i,8):
                w = QtWidgets.QLabel('-')
                self.coinc.append(w)
                g.addWidget(w, i+1, j+1)
        box = QtWidgets.QGroupBox("Group Coincidences")
        box.setLayout(g)
        l.addWidget(box)
        l.addStretch()
        initPvMon(self,f'{pvbase}PATT:COINC',isStruct=True)

        self.setLayout(l)

    def update(self,err):
        if err is None:
            v = self.pv.__value__
            q = v.value.Coinc
            for i,qv in enumerate(q):
                self.coinc[i].setText(str(qv))

def PathTimer(pvbase,parent):

    pathbox = QtWidgets.QWidget()
    pathlo = QtWidgets.QVBoxLayout()
    pathgrid = QtWidgets.QGridLayout()

    textWidgets = []
    for j in range(NGroups):
        ptextWidgets = []
        for i in range(32):
            ptextWidgets.append( PvIntArrayW() )
        textWidgets.append(ptextWidgets)

    parent.ptPvId = []
    pathgrid.addWidget( QtWidgets.QLabel('Group'), 0, 0, 1, 2 )
    for j in range(NGroups):
        pathgrid.addWidget( QtWidgets.QLabel('%d'%j ), 0, j+3 )
        print(f'Creating PathTimer update {pvbase}PART:{j}:PATH_TIME:Update')
        pathgrid.addWidget( PvPushButtonVal(f'{pvbase}PART:{j}:PATH_TIME:Update', 'Upd', 1), 1, j+3 )
    for i in range(14):
        parent.ptPvId.append( PvLinkIdG(pvbase+'RemoteLinkId'+'%d'%i,
                                        pathgrid, i+2, 0) )
        for j in range(NGroups):
            pathgrid.addWidget( textWidgets[j][i], i+2, j+3 )

    parent.pathlnk = []
    for j in range(NGroups):
        ppvbase = f'{pvbase}PART:{j}:PATH_TIME:Array'
        parent.pathlnk.append( PvIntArray( ppvbase, textWidgets[j] ) )

    pathlo.addLayout(pathgrid)
    pathlo.addStretch()
    pathbox.setLayout(pathlo)
    return pathbox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, titles, nopatt):
        global ATCAWidget
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self._pvlabels = []

        tabsel = QtWidgets.QComboBox()
        tabsel.addItems(titles)

        stack = QtWidgets.QStackedWidget()

        for title in titles:
            pvbase = title+':'
            pv = Pv(pvbase+'FwBuild')
            v = pv.get()

            if not isATCA(v):
                amcTitle = 'QSFP'
                nDsLinks = (4,4) # if 'Gen' in v else (3,4)
                ATCAWidget = False
            else:
                amcTitle = 'AMC'
                nDsLinks = (7,7)
                ATCAWidget = True

            tw  = QtWidgets.QTabWidget()

            tb  = QtWidgets.QWidget()
            hl  = QtWidgets.QVBoxLayout()
            #        PvLabel  (hl, pvbase, "PARTITIONS"  )
            #            PvLabel  (hl, pvbase, "PAddr"       , isInt=True)
            PvPAddr  (hl, pvbase, "PAddr"       )
            PvCString(hl, pvbase, "FwBuild"     )
            LblCheckBox(hl, pvbase, "UsRxEnable", enable=False)
            LblCheckBox(hl, pvbase, "CuRxEnable", enable=False)
            LblPushButtonX(hl, pvbase, "ModuleInit"      )
#  These do nothing now
#            LblPushButtonX(hl, pvbase, "DumpPll",        NAmcs)
#            LblPushButtonX(hl, pvbase, "DumpTiming",     2)

#            LblEditIntX   (hl, pvbase, "SetVerbose"      )
            LblPushButtonX(hl, pvbase, "Inhibit"         )
            LblPushButtonX(hl, pvbase, "TagStream"       )
            PvLabel(self, hl, pvbase, "RecClk"     )
            PvLabel(self, hl, pvbase, "FbClk"      )
            PvLabel(self, hl, pvbase, "BpClk"      )
            hl.addStretch()
            tb.setLayout(hl)
            tw.addTab(tb,"Global")

            if 'xtpg' in v:
                tw.addTab( addCuTab (self,pvbase), "CuTiming")
            else:
                tw.addTab( addUsTab (self,pvbase), "UsTiming")

            tw.addTab(FrontPanelAMC(pvbase,nDsLinks[0],          0),f'{amcTitle}0')
            tw.addTab(FrontPanelAMC(pvbase,nDsLinks[1],nDsLinks[0]),f'{amcTitle}1')

            if isATCA(v):
                tw.addTab(PLLs(pvbase,NAmcs),"PLLs")

            tw.addTab(DeadTime(pvbase,self),"DeadTime")

            tw.addTab(GroupsTab(pvbase),"Groups/EventCodes")

            if nopatt==False:
                tw.addTab(PatternTab(pvbase),"Pattern")

            if isATCA(v):
                tw.addTab(PvTableDisplay(pvbase+'SFPSTATUS',[f'Amc{int(j/7)}-{(j%7)}' for j in range(14)]),'SFPs')
            else:
                tw.addTab(PvTableDisplay(pvbase+'QSFPSTATUS',[f'QSFP{int(j/4)}-{(j%4)}' for j in range(8)]),'QSFPs')

            tw.addTab(PathTimer(pvbase,self),"PathTimer")

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
    parser.add_argument("--nopatt", help="no pattern stats", action='store_true')
    parser.add_argument("pvs", help="pvs to monitor",nargs='+')
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.pvs,args.nopatt)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
