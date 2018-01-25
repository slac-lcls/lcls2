
import sys
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot

import pyqtgraph as pg
import subprocess

import process_manager

class Master(object):

    def __init__(self):
        return

    @property
    def detectors(self):
        # would be derived from "the JSON"

        d = {'cspad' : 'area',
             'diode' : 'waveform',
             'opal'  : 'area'}

        return d


class DetectorList(QListWidget):

    def __init__(self, manager, parent=None):
        super(DetectorList, self).__init__(parent)
        self.detectors = {}
        self.itemClicked.connect(self.item_clicked)
        self.manager = manager

    def set_detectors(self, detectors):
        # detectors = dict, maps name --> type
        self.detectors = detectors
        for k in list(detectors.keys()):
            self.addItem(k)
        return

    @pyqtSlot(QListWidgetItem)
    def item_clicked(self, item):

        # WHERE does this enumerated list of detector types come from?

        if self.detectors[item.text()] == 'area':
            # open area detector
            self.manager.get_socket().send_string('string')
            #self._spawn_window('AreaDetector')
            print('create area detector window for:', item.text())

        elif self.detectors[item.text()] == 'waveform':
            self._spawn_window('WaveformDetector')
            print('create waveform window for:', item.text())

        else:
            raise ValueError('Type %s not valid' % self.detectors[item.text()])

        return

    def _spawn_window(self, window_type):

        win = QMainWindow(self)

        if window_type == 'AreaDetector':
            widget = pg.ImageView(win)
            widget.setImage(np.random.randn(1000,1000))

        elif window_type == 'WaveformDetector':
            widget = pg.GraphicsLayoutWidget(win)
            plot_view = widget.addPlot()
            plot_view.plot(np.arange(1000), np.random.randn(1000))

        else:
            raise ValueError('%s not valid window_type' % window_type)

        win.setCentralWidget(widget)
        win.show()

        return win, widget



app = QApplication(sys.argv)
master = Master()
manager = process_manager.ProcessManager()
amilist = DetectorList(manager)

amilist.set_detectors(master.detectors)
amilist.show()

sys.exit(app.exec_())
