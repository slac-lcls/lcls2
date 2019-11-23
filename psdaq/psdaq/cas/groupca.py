import sys
import time
import argparse
from PyQt5 import QtCore, QtGui, QtWidgets
from psdaq.cas.pvedit import *
from psdaq.cas.collection_widget import CollectionWidget

Transitions = [('ClearReadout',0),
               ('Configure'  ,2),
               ('Enable'     ,4),
               ('Disable'    ,5),
               ('Unconfigure',3)]

class PvStateMachine(QtWidgets.QWidget):
    def __init__(self, base, pvbase, xpm, groups):
        super(PvStateMachine,self).__init__()

        print('PvStateMachine',pvbase,groups)

        pvTr      = [tr[0] for tr in Transitions]
        pvTrId    = [tr[1] for tr in Transitions]
        self.trId = pvTrId

        self.pvMsgInsert = Pv(base+':XPM:'+xpm+':GroupMsgInsert')

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
            self.pvMsgHeader .append(Pv(pvbase+g+':MsgHeader'))
            self.pvMsgPayload.append(Pv(pvbase+g+':MsgPayload'))

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

class PvGroupStats(QtWidgets.QWidget):
    def __init__(self, base, pvbase, xpm, groups):
        super(PvGroupStats,self).__init__()

        self.groups = 0
        for g in groups:
            self.groups |= 1<<int(g)

        lo = QtWidgets.QVBoxLayout()

        hlo = QtWidgets.QHBoxLayout()
        b=QtWidgets.QPushButton('Clear')
        b.clicked.connect(self.clear)
        b.setMaximumWidth(45)
        hlo.addWidget(b)
        hlo.addStretch()
        self.pvClear = Pv(base+':XPM:'+xpm+':GroupL0Reset')

        b=QtWidgets.QCheckBox('Run')
        b.toggled.connect(self.run)
        hlo.addWidget(b)
        hlo.addStretch()
        self.pvL0Enable  = Pv(base+':XPM:'+xpm+':GroupL0Enable')
        self.pvL0Disable = Pv(base+':XPM:'+xpm+':GroupL0Disable')
        lo.addLayout(hlo)

        stats = ['L0InpRate','L0AccRate','L1Rate',
                 'RunTime','NumL0Inp','NumL0Acc','NumL1',
                 'DeadFrac','DeadTime']

        glo = QtWidgets.QGridLayout()
        for i,g in enumerate(groups):
            glo.addWidget(QtWidgets.QLabel('Group '+g),0,i+1)
        for i,s in enumerate(stats):
            glo.addWidget(QtWidgets.QLabel(s),i+1,0,QtCore.Qt.AlignRight)
            for j,g in enumerate(groups):
                glo.addWidget(PvDbl(pvbase+g+':'+stats[i]),i+1,j+1)

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

def addGroup(tw, base, group, xpm):
    pvbase = base+group+':'
    wlo    = QtWidgets.QVBoxLayout()

    trgbox = QtWidgets.QGroupBox('Triggers')
    trglo  = QtWidgets.QVBoxLayout()
    LblEditEvt   (trglo, pvbase, "L0Select"        )
    LblEditInt   (trglo, pvbase, "L0Delay"         )
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
    tw.addTab(w,'Group '+group)

#    pvXpm = Pv(pvbase+'XPM')
#    pvXpm.put(int(xpm))
    pvXpm = Pv(pvbase+'Master')
    pvXpm.put(1)

class GroupMaster(QtWidgets.QWidget):
    def __init__(self, pvbase, groups):
        super(GroupMaster,self).__init__()
        hlo = QtWidgets.QHBoxLayout()
        hlo.addWidget(QtWidgets.QLabel('Master:'))
        hlo.addStretch()
        for g in groups:
            b=PvCheckBox(pvbase+g+':Master','Group '+g)
            hlo.addWidget(b)
            hlo.addStretch()
        self.setLayout(hlo)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, base, xpm, groups):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        pvbase = base+':XPM:'+xpm+':PART:'

        print('MainWindow',base,pvbase)

        lol = QtWidgets.QVBoxLayout()

        lol.addWidget(GroupMaster(pvbase,groups))

        tw = QtWidgets.QTabWidget()
        for g in groups:
            addGroup(tw, pvbase, g, xpm)
        tw.addTab(PvStateMachine(base,pvbase,xpm,groups),'Transitions')
        tw.addTab(PvGroupStats  (base,pvbase,xpm,groups),'Events')
        lol.addWidget(tw)

        lol.addStretch()

        ltable = QtWidgets.QWidget()
        ltable.setLayout(lol)

#        partition = int(base.split(':')[-1])
#        print('partition', partition)
#        collectionWidget = CollectionWidget(partition)

        lscroll = QtWidgets.QScrollArea()
        lscroll.setWidget(ltable)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(lscroll)
#        splitter.addWidget(collectionWidget)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(splitter)

        self.centralWidget.setLayout(layout)
        self.centralWidget.resize(440, 660)

        MainWindow.resize(440, 660)
        MainWindow.setWindowTitle(base)
        MainWindow.setCentralWidget(self.centralWidget)

    def master(self,checked):
        if checked:
            self.pvL0Enable.put(self.groups)
            time.sleep(0.1)
            self.pvL0Enable.put(0)
        else:
            self.pvL0Disable.put(self.groups)
            time.sleep(0.1)
            self.pvL0Disable.put(0)

def main():
    print(QtCore.PYQT_VERSION_STR)

#    setCuMode(True)

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,sys.argv[1],sys.argv[2],sys.argv[3:])
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
