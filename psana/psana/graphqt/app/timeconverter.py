#!/usr/bin/env python
"""
Created on 2017-06-14 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-15
"""

from psana.graphqt.QWDateTimeSec import QWDateTimeSec, QApplication, sys

def timeconverter():
    print('Start convertor date and time <-> sec')
    app = QApplication(sys.argv)
    w = QWDateTimeSec(None, show_frame=True, verb=True)
    w.setWindowTitle('Date and time <-> sec')
    w.show()
    app.exec_()
    del app
    sys.exit('End of app')

if __name__ == "__main__":
    timeconverter()

# EOF


