import sys
import argparse
from pvedit import *
import time

NParts = 8

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, base, shelf):
        MainWindow.setObjectName(QtCore.QString.fromUtf8("MainWindow"))
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        lcols = QtGui.QHBoxLayout()

        # Need to wait for pv.get()
        time.sleep(2)

        textWidgets = []
        for i in range(32*NParts):
            textWidgets.append( PvDblArrayW() )

        self.deadflnk = []

        for i in range(NParts):

            pvbase = base+':PART:%d:'%i

            lol = QtGui.QVBoxLayout()

            PvLabel(lol, pvbase, "L0Delay"         )
            PvLabel(lol, pvbase, "Run"             )

            PvLabel(lol, pvbase, "L0InpRate"  )
            PvLabel(lol, pvbase, "L0AccRate"  )
            PvLabel(lol, pvbase, "L1Rate"     )
            PvLabel(lol, pvbase, "RunTime"    )
            PvLabel(lol, pvbase, "NumL0Inp"   )
            PvLabel(lol, pvbase, "NumL0Acc", None, True)
            PvLabel(lol, pvbase, "NumL1"      )
            PvLabel(lol, pvbase, "DeadFrac"   )
            PvLabel(lol, pvbase, "DeadTime"   )

            xbase = base+':XPM:'+shelf+':'
            grid = QtGui.QGridLayout()
            for j in range(14):
                pv = Pv.Pv(xbase+'LinkLabel%d'%j)
                grid.addWidget( QtGui.QLabel(pv.get()), j, 0 )
                grid.addWidget( textWidgets[i*32+j], j, 1 )

            for j in range(16,21):
                pv = Pv.Pv(xbase+'LinkLabel%d'%j)
                grid.addWidget( QtGui.QLabel(pv.get()), j, 0 )
                grid.addWidget( textWidgets[i*32+j], j, 1 )

            for j in range(28,32):
                grid.addWidget( QtGui.QLabel('INH-%d'%(j-28)), j, 0 )
                grid.addWidget( textWidgets[i*32+j], j, 1 )

            self.deadflnk.append( PvDblArray( pvbase+'DeadFLnk', textWidgets[i*32:i*32+31] ) )

            lol.addLayout(grid)

            pw = QtGui.QGroupBox(pvbase)
            pw.setLayout(lol)

            lcols.addWidget(pw)

        ltable = QtGui.QWidget()
        ltable.setLayout(lcols)

        lscroll = QtGui.QScrollArea()
        lscroll.setWidget(ltable)

        layout = QtGui.QHBoxLayout()
        layout.addWidget(lscroll)

        self.centralWidget.setLayout(layout)
        self.centralWidget.resize(640,340)
            
        MainWindow.resize(640,340)
        MainWindow.setWindowTitle(base)
        MainWindow.setCentralWidget(self.centralWidget)

if __name__ == '__main__':
    print(QtCore.PYQT_VERSION_STR)

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("pv", help="pv to monitor")
#    parser.add_argument("shelf", help="shelf")
    args = parser.parse_args()

    app = QtGui.QApplication([])
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
#    ui.setupUi(MainWindow,args.pv)
    ui.setupUi(MainWindow,'DAQ:LAB2','2')
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())
