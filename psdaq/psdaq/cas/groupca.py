import sys
import time
import argparse
from PyQt5 import QtCore, QtGui, QtWidgets
from psdaq.cas.pvedit import *
from psdaq.cas.collection_widget import CollectionWidget
from .xpm_utils import *

Transitions = [('ClearReadout',0),
               ('Configure'  ,2),
               ('Enable'     ,8),
               ('Disable'    ,9),
               ('Unconfigure',3)]

class PvStateMachine(QtWidgets.QWidget):
    def __init__(self, base, pvbase, xpm, groups, prod):
        super(PvStateMachine,self).__init__()

        pvTr      = [tr[0] for tr in Transitions]
        pvTrId    = [tr[1] for tr in Transitions]
        self.trId = pvTrId

        self.pvMsgInsert = Pv(base+':XPM:%d:GroupMsgInsert'%xpm)

        trlo  = QtWidgets.QGridLayout()
        headers = ['Transition','Id','Env']
        for i,h in enumerate(headers):
            trlo.addWidget(QtWidgets.QLabel(h),0,i)
        trlo.addWidget(QtWidgets.QLabel(),0,len(headers))
        trlo.setColumnStretch(len(headers),1)
            
        self.btn = []
        self.env = []
        for i,tr in enumerate(pvTr):
            btn = QtWidgets.QPushButton(tr)
            env = QtWidgets.QLineEdit('0')
            trlo.addWidget(btn                                       ,i+1,0)
            trlo.addWidget(QtWidgets.QLabel('%d'%pvTrId[i])          ,i+1,1)
            trlo.addWidget(env                                       ,i+1,2)
            btn.setCheckable(True)
            btn.toggled.connect(self.transition)
            self.btn.append(btn)
            self.env.append(env)
        trlo.addWidget(QtWidgets.QLabel(),len(pvTr)+1,0)
        trlo.setRowStretch(len(pvTr)+1,1)

        self.group        = 0
        self.pvMsgHeader  = []
        self.pvMsgPayload = []
        for g in groups:
            self.group |= 1<<int(g)
            self.pvMsgHeader .append(Pv(pvbase+'%d:MsgHeader'%g))
            self.pvMsgPayload.append(Pv(pvbase+'%d:MsgPayload'%g))

        self.setLayout(trlo)

    def transition(self,state):
        for i,b in enumerate(self.btn):
            if b.isChecked():
                b.setChecked(False)
                for pv in self.pvMsgHeader:
                    pv.put(self.trId[i])
                for pv in self.pvMsgPayload:
                    pv.put(int(self.env[i].text()))
                self.pvMsgInsert.put(self.group)
                self.pvMsgInsert.put(0)

#  this routine is needed so self.pv.__value__ holds the current result
class PvMonNoCb(object):
    def __init__(self,pvname,isStruct=False):
        initPvMon(self,pvname)

    def update(self,err):
        pass

#  Monitor link group masks and link deadtimes to determine lead bottlenecks
class XpmDeadTime(object):
    def __init__(self,pvbase,xpm,groups,callback):
        self.xpm      = xpm
        self.groups   = groups
        self.callback = callback
        self.dtmax    = {}
        xpmbase = pvbase+':XPM:%d:'%xpm
        self._pv_remoteId  = [ PvMonNoCb(xpmbase+'RemoteLinkId%d'%i) for i in range(14) ]
        self._pv_groupMask = [ PvMonNoCb(xpmbase+'LinkGroupMask%d'%i) for i in range(14) ]
        self._pv_deadFLink = {}
        for g in groups:
            self.dtmax[g] = (None,0.)
            def update(err,self=self,group=g):
                self.update(group)
            self._pv_deadFLink[g] = Pv(xpmbase+'PART:%d:DeadFLnk'%g,update)

    def update(self,group):
        v = self._pv_deadFLink[group].__value__
        dtmax = (None,-1.)
        for i in range(14):
            m = self._pv_groupMask[i].pv.__value__
            if m==(1<<group) and v[i] > dtmax[1]:
                dtmax = (self._pv_remoteId[i].pv.__value__,v[i])
        self.dtmax[group] = dtmax
        self.callback(group)

class DeadTime(object):
    def __init__(self,pvbase,groups,det):
        self.xpm = []
        self.det = det        # dictionary of QLabel widget tuples

        # test for existing xpms
        xnames = [pvbase+':XPM:%d:PAddr'%i for i in range(10)]
        xvalues = pvactx.get(xnames,throw=False)
        for i,v in enumerate(xvalues):
            if isinstance(v,int):
                self.xpm.append(XpmDeadTime(pvbase,i,groups,self.update))

    def update(self,group):
        # Loop through the results from all xpms for this group and update the widget
        dtmax = (None,-1.)
        for xpm in self.xpm:
            if xpm.dtmax[group][1] > dtmax[1]:
                dtmax = xpm.dtmax[group]
        if dtmax[0]:
            names = xpmLinkId(dtmax[0])
            self.det[group][0].setText(names[0]+':'+names[1])        
            self.det[group][1].setText(str(dtmax[1]))
        else:
            self.det[group][0].setText('')
            self.det[group][1].setText('')

class PvGroupStats(QtWidgets.QWidget):
    def __init__(self, base, pvbase, xpm, groups, prod=False):
        super(PvGroupStats,self).__init__()

        self.groups = 0
        for g in groups:
            self.groups |= 1<<g

        lo = QtWidgets.QVBoxLayout()

        hlo = QtWidgets.QHBoxLayout()
        if not prod:
            b=QtWidgets.QPushButton('Clear')
            b.clicked.connect(self.clear)
            b.setMaximumWidth(45)
            hlo.addWidget(b)
            hlo.addStretch()
            self.pvClear = Pv(base+':XPM:%d:GroupL0Reset'%xpm)

        self.runb=QtWidgets.QCheckBox('Run')
        self.runb.setEnabled(not prod)
        if not prod:
            self.runb.clicked.connect(self.run)
        hlo.addWidget(self.runb)
        hlo.addStretch()
        self.pvL0Enable  = Pv(base+':XPM:%d:GroupL0Enable'%xpm ,self.monEnable )
        self.pvL0Disable = Pv(base+':XPM:%d:GroupL0Disable'%xpm,self.monDisable)
        lo.addLayout(hlo)

        #stats = ['L0InpRate','L0AccRate','L1Rate',
        #         'RunTime','NumL0Inp','NumL0Acc','NumL1',
        #         'DeadFrac','DeadTime']
        stats = ['L0InpRate','L0AccRate',
                 'RunTime','NumL0Inp','NumL0Acc',
                 'DeadFrac','DeadTime']

        glo = QtWidgets.QGridLayout()
        for i,g in enumerate(groups):
            glo.addWidget(QtWidgets.QLabel('Group %d'%g),0,i+1)
        for i,s in enumerate(stats):
            glo.addWidget(QtWidgets.QLabel(s),i+1,0,QtCore.Qt.AlignRight)
            for j,g in enumerate(groups):
                glo.addWidget(PvDbl(pvbase+'%d:'%g+stats[i]),i+1,j+1)
        #if prod:
        if True:
            self.det = {}
            for i,g in enumerate(groups):
                self.det[g] = (QtWidgets.QLabel(),QtWidgets.QLabel())
                glo.addWidget(self.det[g][0],len(stats)+i+1,0)
                glo.addWidget(self.det[g][1],len(stats)+i+1,i+1)
            self.deadtime = DeadTime(base,groups,self.det)

        lo.addLayout(glo)
        lo.addStretch()
        self.setLayout(lo)

    def clear(self):
        self.pvClear.put(self.groups)
        self.pvClear.put(0)

    def run(self,checked):
        if checked:
            self.pvL0Enable.put(self.groups)
            time.sleep(0.1)
            self.pvL0Enable.put(0)
        else:
            self.pvL0Disable.put(self.groups)
            time.sleep(0.1)
            self.pvL0Disable.put(0)

    def monEnable(self, err):
        value = self.pvL0Enable.__value__
        if value==self.groups:
            self.runb.setChecked(True)

    def monDisable(self, err):
        value = self.pvL0Disable.__value__
        if value==self.groups:
            self.runb.setChecked(False)

def addGroup(tw, base, group, xpm):
    pvbase = base+'%d:'%group
    wlo    = QtWidgets.QVBoxLayout()

    trgbox = QtWidgets.QGroupBox('Triggers')
    trglo  = QtWidgets.QVBoxLayout()
    LblEditEvt   (trglo, pvbase, "L0Select"        )
    #LblEditInt   (trglo, pvbase, "L0Delay"         )
    LblEditDst   (trglo, pvbase, "DstSelect"       )
    trgbox.setLayout(trglo)
    wlo.addWidget(trgbox)

    inhbox = QtWidgets.QGroupBox('Inhibits')
    inhlo  = QtWidgets.QHBoxLayout()
    LblCheckBox  (inhlo, pvbase, "InhEnable"  , 4, horiz=False  )
    LblEditInt   (inhlo, pvbase, "InhInterval", 4, horiz=False  )
    LblEditInt   (inhlo, pvbase, "InhLimit"   , 4, horiz=False  )
    inhbox.setLayout(inhlo)
    wlo.addWidget(inhbox)

    w = QtWidgets.QWidget()
    w.setLayout(wlo)
    tw.addTab(w,'Group %d'%group)

    pvXpm = Pv(pvbase+'Main')
    pvXpm.put(1)

class GroupMain(QtWidgets.QWidget):
    def __init__(self, pvbase, groups):
        super(GroupMain,self).__init__()
        hlo = QtWidgets.QHBoxLayout()
        hlo.addWidget(QtWidgets.QLabel('Main:'))
        hlo.addStretch()
        for g in groups:
            b=PvCheckBox(pvbase+'%d:Main'%g,'Group %d'%g)
            hlo.addWidget(b)
            hlo.addStretch()
        self.setLayout(hlo)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, base, xpm, groups, prod):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        pvbase = base+':XPM:%d:PART:'%xpm

        lol = QtWidgets.QVBoxLayout()

        if not prod:
            lol.addWidget(GroupMain(pvbase,groups))

            tw = QtWidgets.QTabWidget()
            for g in groups:
                addGroup(tw, pvbase, g, xpm)
            tw.addTab(PvStateMachine(base,pvbase,xpm,groups,prod),'Transitions')
            tw.addTab(PvGroupStats  (base,pvbase,xpm,groups),'Events')
            tw.setCurrentIndex(len(groups)+1)
            lol.addWidget(tw)
        else:
            lol.addWidget(PvGroupStats(base,pvbase,xpm,groups,prod))

        lol.addStretch()

        ltable = QtWidgets.QWidget()
        ltable.setLayout(lol)

        lscroll = QtWidgets.QScrollArea()
        lscroll.setWidget(ltable)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(lscroll)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(splitter)

        self.centralWidget.setLayout(layout)
        self.centralWidget.resize(440, 660)

        MainWindow.resize(440, 660)
        MainWindow.setWindowTitle(base)
        MainWindow.setCentralWidget(self.centralWidget)

    def main(self,checked):
        if checked:
            self.pvL0Enable.put(self.groups)
            time.sleep(0.1)
            self.pvL0Enable.put(0)
        else:
            self.pvL0Disable.put(self.groups)
            time.sleep(0.1)
            self.pvL0Disable.put(0)

def main():
    parser = argparse.ArgumentParser(description='Readout group control and monitoring')
    parser.add_argument('--prod', help='Production Mode', action='store_true', default=False)
    parser.add_argument('pvbase', help='EPICS PV Prefix', type=str)
    parser.add_argument('xpmroot', help='XPM at root', type=int)
    parser.add_argument('groups' , help='list of groups', type=int, nargs='+')
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.pvbase,args.xpmroot,args.groups,args.prod)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
