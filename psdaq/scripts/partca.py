import sys
import argparse
from pvedit import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, base):
        MainWindow.setObjectName(QtCore.QString.fromUtf8("MainWindow"))
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        pvbase = base+':'

        lol = QtGui.QVBoxLayout()

        trgbox = QtGui.QGroupBox('Trigger')
        trglo = QtGui.QVBoxLayout()
        LblEditEvt   (trglo, pvbase, "L0Select"        )
        LblEditInt   (trglo, pvbase, "L0Delay"         )
#        LblEditDst   (trglo, pvbase, "DstSelect"       )
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

        msgbox = QtGui.QGroupBox('Message')
        msglo  = QtGui.QHBoxLayout()
        msglo.addWidget(PvPushButton(pvbase+"MsgInsert","Insert"))
        msglo.addWidget(PvEditInt(pvbase+"MsgHeader","Hdr"))
        msglo.addWidget(PvEditInt(pvbase+"MsgPayload","Payload"))
        msglo.addStretch()
        msgbox.setLayout(msglo)
        lol.addWidget(msgbox)

        inhbox = QtGui.QGroupBox('Inhibits')
        inhlo = QtGui.QVBoxLayout()
        LblEditInt   (inhlo, pvbase, "InhInterval", 4  )
        LblEditInt   (inhlo, pvbase, "InhLimit"   , 4  )
        LblCheckBox  (inhlo, pvbase, "InhEnable"  , 4  )
        inhbox.setLayout(inhlo)
        lol.addWidget(inhbox)

        #lol.addStretch()

        lor = QtGui.QVBoxLayout()
        
        b=PvPushButton(pvbase+'ResetL0', "Clear")
        b.setMaximumWidth(45)
        lor.addWidget(b)
        PvLabel(lor, pvbase, "L0InpRate"  )
        PvLabel(lor, pvbase, "L0AccRate"  )
        PvLabel(lor, pvbase, "L1Rate"     )
        PvLabel(lor, pvbase, "RunTime"    )
        PvLabel(lor, pvbase, "NumL0Inp"   )
        PvLabel(lor, pvbase, "NumL0Acc", None, True)
        PvLabel(lor, pvbase, "NumL1"      )
        PvLabel(lor, pvbase, "DeadFrac"   )
        PvLabel(lor, pvbase, "DeadTime"   )

        ltable = QtGui.QWidget()
        ltable.setLayout(lol)
        rtable = QtGui.QWidget()
        rtable.setLayout(lor)

        lscroll = QtGui.QScrollArea()
        lscroll.setWidget(ltable)
        rscroll = QtGui.QScrollArea()
        rscroll.setWidget(rtable)

        splitter = QtGui.QSplitter()
        splitter.addWidget(lscroll)
        splitter.addWidget(rscroll)

        layout = QtGui.QHBoxLayout()
        layout.addWidget(splitter)

        self.centralWidget.setLayout(layout)
        self.centralWidget.resize(640,340)
            
        MainWindow.resize(640,340)
        MainWindow.setWindowTitle(base)
        MainWindow.setCentralWidget(self.centralWidget)

if __name__ == '__main__':
    print QtCore.PYQT_VERSION_STR

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("pv", help="pv to monitor")
    args = parser.parse_args()

    app = QtGui.QApplication([])
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow,args.pv)
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())
