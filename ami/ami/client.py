import zmq
import sys
import time
import json
import argparse
import threading
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, QTimer

import pyqtgraph as pg

from ami.comm import ZmqPorts, ZmqConfig, ZmqBase
from ami.data import DataTypes
from ami.operation import ROI


class Master(ZmqBase):

    def __init__(self, zmqctx, zmq_config):
        super(__class__, self).__init__("client-master", zmq_config, zmqctx)
        self.sock = self.connect(zmq.REQ)

    @property
    def graph(self):
        self.sock.send_string('get_graph')
        return self.sock.recv_pyobj()

    @property
    def features(self):
        self.sock.send_string('get_features')
        return self.sock.recv_pyobj()

    def update(self, graph):
        self.sock.send_string('set_graph', zmq.SNDMORE)
        self.sock.send_pyobj(graph)
        if self.sock.recv_string() == 'ok':
            self.sock.send_string('apply_graph')
            return self.sock.recv_string() == 'ok'
        else:
            return False


class WaveformWidget(pg.GraphicsLayoutWidget):
    def __init__(self, topic, master, timer, parent=None):
        super(WaveformWidget, self).__init__(parent)
        self.plot_view = self.addPlot()
        self.plot = None
        self.master = master
        self.topic = topic
        self.timer = timer
        self.sock = self.master.connect(zmq.REQ)
        self.timer.timeout.connect(self.get_waveform)

    @pyqtSlot()
    def get_waveform(self):
        self.sock.send_string("feature:%s"%self.topic)
        reply = self.sock.recv_string()
        if reply == 'ok':
            self.waveform_updated(self.sock.recv_pyobj())
        else:
            print("failed to fetch %s from manager!"%self.topic)

    def waveform_updated(self, data):
        if self.plot is None:
            self.plot = self.plot_view.plot(np.arange(data.size), data)
        else:
            self.plot.setData(y=data)



class AreaDetWidget(pg.ImageView):
    def __init__(self, topic, master, timer, parent=None):
        super(AreaDetWidget, self).__init__(parent)
        self.master = master
        self.topic = topic
        self.timer = timer
        self.sock = self.master.connect(zmq.REQ)
        self.timer.timeout.connect(self.get_image)
        self.roi.sigRegionChangeFinished.connect(self.roi_updated)
    
    @pyqtSlot()
    def get_image(self):
        self.sock.send_string("feature:%s"%self.topic)
        reply = self.sock.recv_string()
        if reply == 'ok':
            self.image_updated(self.sock.recv_pyobj())
        else:
            print("failed to fetch %s from manager!"%self.topic)

    def image_updated(self, data):
        self.setImage(data)

    #@pyqtSlot(pg.ROI)
    def roi_updated(self, roi):
        graph = self.master.graph
        roi = ROI(*roi.getAffineSliceParams(self.image, self.getImageItem()), (0,1))
        graph["%s-roi"%self.topic] = { "optype": "ROI", "config": roi.export(), "inputs": [{"name": self.topic, "required": True}] }
        self.master.update(graph)
        

class DetectorList(QListWidget):
    
    def __init__(self, parent=None, zmqctx=None, zmqcfg=None):
        super(DetectorList, self).__init__(parent)
        self.master = Master(zmqctx, zmqcfg)
        self.features = {}
        self.timer = QTimer()
        self.timer.timeout.connect(self.get_features)
        self.timer.start(1000)
        self.itemClicked.connect(self.item_clicked)
        return

    def load(self, graph_cfg):
        self.master.update(graph_cfg)

    @pyqtSlot()
    def get_features(self):
        # detectors = dict, maps name --> type
        self.features = self.master.features
        self.clear()
        for k in self.features.keys():
            self.addItem(k)
        return

    @pyqtSlot(QListWidgetItem)
    def item_clicked(self, item):

        # WHERE does this enumerated list of detector types come from?

        if self.features[item.text()] == DataTypes.Image:
            self._spawn_window('AreaDetector', item.text())
            print('create area detector window for:', item.text())

        elif self.features[item.text()] == DataTypes.Waveform:
            self._spawn_window('WaveformDetector', item.text())
            print('create waveform window for:', item.text())

        else:
            raise ValueError('Type %s not valid' % self.detectors[item.text()])

        return

    def _spawn_window(self, window_type, topic):
        
        win = QMainWindow(self)

        if window_type == 'AreaDetector':
            widget = AreaDetWidget(topic, self.master, self.timer, win)
        
        elif window_type == 'WaveformDetector':
            widget = WaveformWidget(topic, self.master, self.timer, win)
        
        else:
            raise ValueError('%s not valid window_type' % window_type)
        
        win.setCentralWidget(widget)
        win.setWindowTitle(topic)
        win.show()

        return win, widget

def main():
    parser = argparse.ArgumentParser(description='AMII GUI Client')

    parser.add_argument(
        '-H',
        '--host',
        default='localhost',
        help='hostname of the AMII Manager'
    )

    parser.add_argument(
        '-p',
        '--platform',
        type=int,
        default=0,
        help='platform number of the AMII - selects port range to use (default: 0)'
    )

    parser.add_argument(
        '-l',
        '--load',
        help='saved AMII configuration to load'
    )

    args = parser.parse_args()

    saved_cfg = None
    if args.load is not None:
        try:
            with open(args.load, 'r') as cnf:
                saved_cfg = json.load(cnf)
        except OSError as os_exp:
            print("ami-client: problem opening saved graph configuration file:", os_exp)
            return 1
        except json.decoder.JSONDecodeError as json_exp:
            print("ami-client: problem parsing saved graph configuration file (%s):"%args.load, json_exp)
            return 1

    zmqcfg = ZmqConfig(
        args.platform,
        binds={},
        connects={zmq.REQ: (args.host, ZmqPorts.Command)}
    )

    try:
        ctx = zmq.Context()
        app = QApplication(sys.argv)
        amilist = DetectorList(zmqctx=ctx, zmqcfg=zmqcfg)
        if saved_cfg is not None:
            amilist.load(saved_cfg)
        amilist.show()

        return app.exec_()
    except KeyboardInterrupt:
        print("Client killed by user...")
        return 0


if __name__ == '__main__':
    sys.exit(main())
