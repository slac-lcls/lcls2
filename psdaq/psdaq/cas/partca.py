import sys
import argparse
from PyQt5 import QtCore, QtGui, QtWidgets
from psdaq.cas.pvedit import *
from psdaq.cas.collection_widget import CollectionWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, base):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        pvbase = base+':'

        lol = QtWidgets.QVBoxLayout()

        LblEditInt   (lol, pvbase, 'XPM')
#        lol.addWidget( PvEditInt(pvbase+'XPM','XPM') );

        trgbox = QtWidgets.QGroupBox('Trigger')
        trglo = QtWidgets.QVBoxLayout()
        LblEditEvt   (trglo, pvbase, "L0Select"        )
        LblEditInt   (trglo, pvbase, "L0Delay"         )
        LblEditDst   (trglo, pvbase, "DstSelect"       )
        LblCheckBox  (trglo, pvbase, "Run"             )
        trgbox.setLayout(trglo)
        lol.addWidget(trgbox)
#        LblCheckBox  (lol, pvbase, "ClearStats"      )

#        LblCheckBox  (lol, pvbase, "L1TrgClear",     NPartitions)
#        LblCheckBox  (lol, pvbase, "L1TrgEnable",    NPartitions)
#        LblEditTS    (lol, pvbase, "L1TrgSource",    NPartitions)
#        LblEditInt   (lol, pvbase, "L1TrgWord",      NPartitions)
#        LblCheckBox  (lol, pvbase, "L1TrgWrite",     NPartitions)

#        LblEditInt   (lol, pvbase, "AnaTagReset",    NPartitions)
#        LblEditInt   (lol, pvbase, "AnaTag",         NPartitions)
#        LblEditInt   (lol, pvbase, "AnaTagPush",     NPartitions)

#        LblEditInt   (lol, pvbase, "PipelineDepth")

        trbox = QtWidgets.QGroupBox('Transitions')
        trlo  = QtWidgets.QVBoxLayout()
        cfglo = QtWidgets.QHBoxLayout()
        cfglo.addWidget(PvPushButton(pvbase+"MsgConfig","Configure"))
        cfglo.addWidget(PvEditInt   (pvbase+"MsgConfigKey","Key"))
        trlo.addLayout(cfglo)
        trlo.addWidget(PvPushButton(pvbase+"MsgEnable"     ,"Enable"))
        trlo.addWidget(PvPushButton(pvbase+"MsgDisable"    ,"Disable"))
        trlo.addWidget(PvPushButton(pvbase+"MsgClear" ,"ClearReadout"))
        trbox.setLayout(trlo)
        lol.addWidget(trbox)

        msglo  = QtWidgets.QHBoxLayout()
        msgbox = QtWidgets.QGroupBox('Message')
        msglo  = QtWidgets.QHBoxLayout()
        msglo.addWidget(PvPushButton(pvbase+"MsgInsert","Insert"))
        msglo.addWidget(PvEditInt   (pvbase+"MsgHeader","Hdr"))
        msglo.addWidget(PvEditInt   (pvbase+"MsgPayload","Payload"))
        msglo.addStretch()
        msgbox.setLayout(msglo)
        lol.addWidget(msgbox)

        inhbox = QtWidgets.QGroupBox('Inhibits')
        inhlo = QtWidgets.QHBoxLayout()
        LblCheckBox  (inhlo, pvbase, "InhEnable"  , 4, horiz=False  )
        LblEditInt   (inhlo, pvbase, "InhInterval", 4, horiz=False  )
        LblEditInt   (inhlo, pvbase, "InhLimit"   , 4, horiz=False  )
        inhbox.setLayout(inhlo)
        lol.addWidget(inhbox)

        #lol.addStretch()

        lor = QtWidgets.QVBoxLayout()

        b=PvPushButton(pvbase+'ResetL0', "Clear")
        b.setMaximumWidth(45)
        lor.addWidget(b)
        PvLabel(lor, pvbase, "L0InpRate", scale=1.0  )
        PvLabel(lor, pvbase, "L0AccRate", scale=1.0  )
        PvLabel(lor, pvbase, "L1Rate"     )
        PvLabel(lor, pvbase, "RunTime"    )
        PvLabel(lor, pvbase, "NumL0Inp"   )
        PvLabel(lor, pvbase, "NumL0Acc", None, True)
        PvLabel(lor, pvbase, "NumL1"      )
        PvLabel(lor, pvbase, "DeadFrac", scale=1.0   )
        PvLabel(lor, pvbase, "DeadTime", scale=1.0   )

        ltable = QtWidgets.QWidget()
        ltable.setLayout(lol)
        rtable = QtWidgets.QWidget()
        rtable.setLayout(lor)

        partition = int(base.split(':')[-1])
        print('partition', partition)
        collectionWidget = CollectionWidget(partition)

        lscroll = QtWidgets.QScrollArea()
        lscroll.setWidget(ltable)
        rscroll = QtWidgets.QScrollArea()
        rscroll.setWidget(rtable)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(lscroll)
        splitter.addWidget(rscroll)
        splitter.addWidget(collectionWidget)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(splitter)

        self.centralWidget.setLayout(layout)
        self.centralWidget.resize(740, 340)
        MainWindow.resize(740, 340)
        MainWindow.setWindowTitle(base)
        MainWindow.setCentralWidget(self.centralWidget)

def main():
    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    parser.add_argument("pv", help="pv to monitor")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.pv)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
