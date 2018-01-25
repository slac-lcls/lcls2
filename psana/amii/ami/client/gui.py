import zmq
import sys
import threading
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot

import pyqtgraph as pg


class Master(object):

    def __init__(self, zmqctx, host, port):
        self.ctx = zmqctx
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect("tcp://%s:%d"%(host, port))


    @property
    def detectors(self):
        d = {}
        self.sock.send_string('get_config')
        cfg = self.sock.recv_pyobj()
        for topic, shape in cfg['src']['config']['sources']:
            if len(shape) > 1:
                d[topic] = 'area'
            else:
                d[topic] = 'waveform'
        # would be derived from "the JSON"

        #d = {'cspad' : 'area',
        #     'diode' : 'waveform',
        #     'opal'  : 'area'}

        return d


class WaveformWidget(pg.GraphicsLayoutWidget):
    def __init__(self, topic, zmqctx, parent=None):
        super(WaveformWidget, self).__init__(parent)
        self.plot_view = self.addPlot()
        self.plot = None
        if zmqctx is None:
            self.ctx = zmq.Context()
        else:
            self.ctx = zmqctx
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, topic.decode('utf-8'))
        self.sock.connect("tcp://localhost:55558")
        self.zmq_thread = threading.Thread(target=self.get_waveform)
        self.zmq_thread.daemon = True

    def get_waveform(self):
        while True:
            self.sock.recv_string()
            self.waveform_updated(self.sock.recv_pyobj())

    def waveform_updated(self, data):
        if self.plot is None:
            self.plot = self.plot_view.plot(np.arange(data.size), data)
        else:
            self.plot.setData(y=data)



class AreaDetWidget(pg.ImageView):
    def __init__(self, topic, zmqctx, parent=None):
        super(AreaDetWidget, self).__init__(parent)
        if zmqctx is None:
            self.ctx = zmq.Context()
        else:
            self.ctx = zmqctx
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, topic.decode('utf-8'))
        self.sock.connect("tcp://localhost:55556")
        self.zmq_thread = threading.Thread(target=self.get_image)
        self.zmq_thread.daemon = True
        self.roi.sigRegionChangeFinished.connect(self.roi_updated)
    
    def get_image(self):
        while True:
            self.sock.recv_string()
            self.image_updated(self.sock.recv_pyobj())

    def image_updated(self, data):
        self.setImage(data)

    #@pyqtSlot(pg.ROI)
    def roi_updated(self, roi):
        print(roi.getAffineSliceParams(self.image, self.getImageItem()))

class DetectorList(QListWidget):
    
    def __init__(self, parent=None, zmqctx=None):
        super(DetectorList, self).__init__(parent)
        self.detectors = {}
        self.ctx = zmqctx
        self.itemClicked.connect(self.item_clicked)
        return

    def set_detectors(self, detectors):
        # detectors = dict, maps name --> type
        self.detectors = detectors
        for k in detectors.keys():
            self.addItem(k)
        return

    @pyqtSlot(QListWidgetItem)
    def item_clicked(self, item):

        # WHERE does this enumerated list of detector types come from?

        if self.detectors[item.text()] == 'area':
            # open area detector
            self._spawn_window('AreaDetector', item.text(), self.ctx)
            print 'create area detector window for:', item.text()

        elif self.detectors[item.text()] == 'waveform':
            self._spawn_window('WaveformDetector', item.text(), self.ctx)
            print 'create waveform window for:', item.text()

        else:
            raise ValueError('Type %s not valid' % self.detectors[item.text()])

        return

    def _spawn_window(self, window_type, topic, zmqctx):
        
        win = QMainWindow(self)

        if window_type == 'AreaDetector':
            widget = AreaDetWidget(topic, zmqctx, win)
            widget.zmq_thread.start()
        
        elif window_type == 'WaveformDetector':
            widget = WaveformWidget(topic, zmqctx, win)
            widget.zmq_thread.start()
        
        else:
            raise ValueError('%s not valid window_type' % window_type)
        
        win.setCentralWidget(widget)
        win.show()

        return win, widget
        

ctx = zmq.Context()
app = QApplication(sys.argv)
master  = Master(ctx, 'localhost', 5555)
amilist = DetectorList(zmqctx=ctx)

amilist.set_detectors(master.detectors)
amilist.show()

sys.exit(app.exec_())


