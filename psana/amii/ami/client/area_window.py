from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys
import zmq
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot
import numpy as np
import pyqtgraph as pg

port = sys.argv[1]


context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect('tcp://localhost:%s' %port)

app = QApplication(sys.argv)

print(socket.recv_string())
window = QMainWindow()
widget = pg.ImageView()
widget.setImage(np.random.randn(1000,1000))
window.setCentralWidget(widget)
window.show()

app.exec_()
print('finished')
