import sys
import argparse
from pvedit import *
import time

NParts = 8

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, base, shelf):
        MainWindow.setObjectName("MainWindow")
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        lcols = QtWidgets.QHBoxLayout()

        # Need to wait for pv.get()
        time.sleep(2)

        textWidgets = []
        for i in range(32*NParts):
            textWidgets.append( PvDblArrayW() )

        self.deadflnk = []

        for i in range(NParts):

            pvbase = base+':PART:%d:'%i

            lol = QtWidgets.QVBoxLayout()

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
            grid = QtWidgets.QGridLayout()
            for j in range(14):
                pv = Pv.Pv(xbase+'LinkLabel%d'%j)
                grid.addWidget( QtWidgets.QLabel(pv.get()), j, 0 )
                grid.addWidget( textWidgets[i*32+j], j, 1 )

            for j in range(16,21):
                pv = Pv.Pv(xbase+'LinkLabel%d'%j)
                grid.addWidget( QtWidgets.QLabel(pv.get()), j, 0 )
                grid.addWidget( textWidgets[i*32+j], j, 1 )

            for j in range(28,32):
                grid.addWidget( QtWidgets.QLabel('INH-%d'%(j-28)), j, 0 )
                grid.addWidget( textWidgets[i*32+j], j, 1 )

            self.deadflnk.append( PvDblArray( pvbase+'DeadFLnk', textWidgets[i*32:i*32+32] ) )

            lol.addLayout(grid)

            pw = QtWidgets.QGroupBox(pvbase)
            pw.setLayout(lol)

            lcols.addWidget(pw)

        ltable = QtWidgets.QWidget()
        ltable.setLayout(lcols)

        lscroll = QtWidgets.QScrollArea()
        lscroll.setWidget(ltable)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(lscroll)

        self.centralWidget.setLayout(layout)
        self.centralWidget.resize(640,340)
            
        MainWindow.resize(640,340)
        MainWindow.setWindowTitle(base)
        MainWindow.setCentralWidget(self.centralWidget)

if __name__ == '__main__':
    print(QtCore.PYQT_VERSION_STR)

#    parser = argparse.ArgumentParser(description='simple pv monitor gui')
#    parser.add_argument("pv", help="pv to monitor")
#    parser.add_argument("shelf", help="shelf")
#    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
#    ui.setupUi(MainWindow,args.pv)
    ui.setupUi(MainWindow,'DAQ:LAB2','2')
    MainWindow.updateGeometry()

    MainWindow.show()
    sys.exit(app.exec_())
