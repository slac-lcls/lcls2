import re
import zmq
import sys
import time
import json
import argparse
import threading
import numpy as np
import multiprocessing as mp

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, QTimer, QRect

import pyqtgraph as pg

from ami.data import DataTypes
from ami.comm import Ports
from ami.operation import ROINode, EvalNode


class CommunicationHandler(object):

    def __init__(self, host, port):
        self.ctx  = zmq.Context()
        self.port = port
        self.host = host
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect('tcp://%s:%d' % (self.host, self.port))

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
        return self.sock.recv_string() == 'ok'


class ScalarWidget(QLCDNumber):
    def __init__(self, topic, host, port, parent=None):
        super(__class__, self).__init__(parent)
        self.topic = topic
        self.timer = QTimer()
        self.setGeometry(QRect(320, 180, 191, 81))
        self.setDigitCount(10)
        self.setObjectName(topic)
        self.comm_handler = CommunicationHandler(host, port)
        self.timer.timeout.connect(self.get_scalar)
        self.timer.start(1000)

    @pyqtSlot()
    def get_scalar(self):
        self.comm_handler.sock.send_string("feature:%s"%self.topic)
        reply = self.comm_handler.sock.recv_string()
        if reply == 'ok':
            self.scalar_updated(self.comm_handler.sock.recv_pyobj())
        else:
            print("failed to fetch %s from manager!"%self.topic)

    def scalar_updated(self, data):
        self.display(data)


class WaveformWidget(pg.GraphicsLayoutWidget):
    def __init__(self, topic, host, port, parent=None):
        super(__class__, self).__init__(parent)
        self.topic = topic
        self.timer = QTimer()
        self.comm_handler = CommunicationHandler(host, port)
        self.plot_view = self.addPlot()
        self.plot = None
        self.timer.timeout.connect(self.get_waveform)
        self.timer.start(1000)

    @pyqtSlot()
    def get_waveform(self):
        self.comm_handler.sock.send_string("feature:%s"%self.topic)
        reply = self.comm_handler.sock.recv_string()
        if reply == 'ok':
            self.waveform_updated(self.comm_handler.sock.recv_pyobj())
        else:
            print("failed to fetch %s from manager!"%self.topic)

    def waveform_updated(self, data):
        if self.plot is None:
            self.plot = self.plot_view.plot(np.arange(data.size), data)
        else:
            self.plot.setData(y=data)


class AreaDetWidget(pg.ImageView):
    def __init__(self, topic, host, port, parent=None):
        super(AreaDetWidget, self).__init__(parent)
        self.topic = topic
        self.comm_handler = CommunicationHandler(host, port)
        self.timer = QTimer()
        self.timer.timeout.connect(self.get_image)
        self.timer.start(1000)
        self.roi.sigRegionChangeFinished.connect(self.roi_updated)
    
    @pyqtSlot()
    def get_image(self):
        self.comm_handler.sock.send_string("feature:%s"%self.topic)
        reply = self.comm_handler.sock.recv_string()
        if reply == 'ok':
            self.image_updated(self.comm_handler.sock.recv_pyobj())
        else:
            print("failed to fetch %s from manager!"%self.topic)

    def image_updated(self, data):
        self.setImage(data)

    #@pyqtSlot(pg.ROI)
    def roi_updated(self, roi):
        graph = self.comm_handler.graph
        roi = ROINode("%s-roi"%self.topic, *roi.getAffineSliceParams(self.image, self.getImageItem()), (0,1), self.topic)
        graph[roi.name] = roi.export()
        self.comm_handler.update(graph)
        

class Calculator(QWidget):
    def __init__(self, comm, parent=None):
        super(Calculator, self).__init__(parent)
        self.setWindowTitle("Calculator")
        self.comm = comm
        self.move(280, 80)
        self.resize(280,40)
        self.field_parse = re.compile("\s+")

        self.nameLabel = QLabel('Name:', self)
        self.nameBox = QLineEdit(self)
        self.inputsLabel = QLabel('Inputs:', self)
        self.inputsBox = QLineEdit(self)
        self.importsLabel = QLabel('Imports:', self)
        self.importsBox = QLineEdit(self)
        self.codeLabel = QLabel('Expression:', self)
        self.codeBox = QLineEdit(self)
        self.button = QPushButton('Apply', self)
        self.button.clicked.connect(self.on_click)

        self.calc_layout = QVBoxLayout(self)
        self.calc_layout.addWidget(self.nameLabel)
        self.calc_layout.addWidget(self.nameBox)
        self.calc_layout.addWidget(self.inputsLabel)
        self.calc_layout.addWidget(self.inputsBox)
        self.calc_layout.addWidget(self.importsLabel)
        self.calc_layout.addWidget(self.importsBox)
        self.calc_layout.addWidget(self.codeLabel)
        self.calc_layout.addWidget(self.codeBox)
        self.calc_layout.addWidget(self.button)
        self.setLayout(self.calc_layout)

    def parse_inputs(self):
        if self.inputsBox.text():
            return self.field_parse.split(self.inputsBox.text())
        else:
            return []

    def parse_imports(self):
        if self.importsBox.text():
            return self.field_parse.split(self.importsBox.text())
        else:
            return None

    def make_graph_node(self):
        node = EvalNode(self.importsBox.text(), self.codeBox.text(), *self.parse_inputs())
        imports = self.parse_imports()
        if imports is not None:
            node.imports(*imports)
        return node.export()

    @pyqtSlot()
    def on_click(self):
        graph = self.comm.graph
        graph[self.nameBox.text()] = self.make_graph_node()
        self.comm.update(graph)

class DetectorList(QListWidget):
    
    def __init__(self, queue, comm_handler, parent=None):
        super(DetectorList, self).__init__(parent)
        self.queue = queue
        self.comm_handler = comm_handler
        self.features = {}
        self.timer = QTimer()
        self.timer.timeout.connect(self.get_features)
        self.timer.start(1000)
        self.itemClicked.connect(self.item_clicked)
        return

    def load(self, graph_cfg):
        self.comm_handler.update(graph_cfg)

    @pyqtSlot()
    def get_features(self):
        # detectors = dict, maps name --> type
        self.features = self.comm_handler.features
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

        elif self.features[item.text()] == DataTypes.Scalar:
            self._spawn_window('ScalarDetector', item.text())
            print('create waveform window for:', item.text())

        else:
            print('Type %s not valid' % self.features[item.text()])

        return

    def _spawn_window(self, window_type, topic):
        self.queue.put((window_type, topic))
        

def run_list_window(queue, host, port, ami_save):
    app = QApplication(sys.argv)
    comm_handler = CommunicationHandler(host, port)
    amilist = DetectorList(queue, comm_handler)
    if ami_save is not None:
        amilist.load(ami_save)
    amilist.show()
    calc = Calculator(comm_handler)
    calc.show()

    # wait for the qt app to exit
    retval = app.exec_()

    # send exit signal to master process
    queue.put(("exit", None))

    return retval


def run_widget(queue, window_type, topic, host, port):

    ctx = zmq.Context()
    app = QApplication(sys.argv)
    win = QMainWindow()

    if window_type == 'AreaDetector':
        widget = AreaDetWidget(topic, host, port, win)

    elif window_type == 'WaveformDetector':
        widget = WaveformWidget(topic, host, port, win)

    elif window_type == 'ScalarDetector':
        widget = ScalarWidget(topic, host, port, win)

    else:
        raise ValueError('%s not valid window_type' % window_type)

    win.setCentralWidget(widget)
    win.setWindowTitle(topic)
    win.show()

    return app.exec_()


def main():
    parser = argparse.ArgumentParser(description='AMII GUI Client')

    parser.add_argument(
        '-H',
        '--host',
        default='localhost',
        help='hostname of the AMII Manager (default: localhost)'
    )

    parser.add_argument(
        '-p',
        '--port',
        type=int,
        default=Ports.Comm,
        help='port for manager/client (GUI) communication (default: %d)'%Ports.Comm
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


    try:
        queue = mp.Queue()
        list_proc = mp.Process(target=run_list_window, args=(queue, args.host, args.port, saved_cfg))
        list_proc.start()
        widget_procs = []

        while True:
            window_type, topic = queue.get()
            if window_type == 'exit':
                print("received exit signal - exiting!")
                break;
            print("opening new widget:", window_type, topic)
            proc = mp.Process(target=run_widget, args=(queue, window_type, topic, args.host, args.port))
            proc.start()
            widget_procs.append(proc)
    except KeyboardInterrupt:
        print("Client killed by user...")
        return 0


if __name__ == '__main__':
    sys.exit(main())
