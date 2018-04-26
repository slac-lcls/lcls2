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
from ami.graph import Graph


class CommunicationHandler(object):

    def __init__(self, addr):
        self.ctx  = zmq.Context()
        self.addr = addr
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(self.addr)

    @property
    def graph(self):
        self.sock.send_string('get_graph')
        return self.sock.recv_pyobj()

    @property
    def features(self):
        self.sock.send_string('get_features')
        return self.sock.recv_pyobj()

    def clear(self):
        self.sock.send_string('clear_graph')
        return self.sock.recv_string() == 'ok'

    def reset(self):
        self.sock.send_string('reset_features')
        return self.sock.recv_string() == 'ok'

    def update(self, graph):
        self.sock.send_string('set_graph', zmq.SNDMORE)
        self.sock.send_pyobj(graph)
        return self.sock.recv_string() == 'ok'


class ScalarWidget(QLCDNumber):
    def __init__(self, topic, addr, parent=None):
        super(__class__, self).__init__(parent)
        self.topic = topic
        self.timer = QTimer()
        self.setGeometry(QRect(320, 180, 191, 81))
        self.setDigitCount(10)
        self.setObjectName(topic)
        self.comm_handler = CommunicationHandler(addr)
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
    def __init__(self, topic, addr, parent=None):
        super(__class__, self).__init__(parent)
        self.topic = topic
        self.timer = QTimer()
        self.comm_handler = CommunicationHandler(addr)
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
    def __init__(self, topic, addr, parent=None):
        super(AreaDetWidget, self).__init__(parent)
        self.topic = topic
        self.comm_handler = CommunicationHandler(addr)
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
        shape, vector, origin = roi.getAffineSliceParams(self.image, self.getImageItem())
        config = {
            "shape": shape,
            "vector": vector,
            "origin": origin,
            "axes": (0,1),
        }
        roi = Graph.build_node(
            "{0:s}_roi = pg.affineSlice({0:s}, config['shape'], config['origin'], config['vector'], config['axes'])".format(self.topic),
            self.topic,
            "%s_roi"%self.topic,
            config=config,
            imports=[('pyqtgraph', 'pg')],
        )
        graph["%s_roi"%self.topic] = roi
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
            return [(imp, imp) for imp in self.field_parse.split(self.importsBox.text())]
        else:
            return None

    @pyqtSlot()
    def on_click(self):
        graph = self.comm.graph
        graph[self.nameBox.text()] = Graph.build_node(
            "%s = %s"%(self.nameBox.text(), self.codeBox.text()),
            self.parse_inputs(),
            self.nameBox.text(),
            imports=self.parse_imports()
        )
        self.comm.update(graph)

class DetectorList(QListWidget):
    
    def __init__(self, queue, comm_handler, parent=None):
        super(__class__, self).__init__(parent)
        self.queue = queue
        self.comm_handler = comm_handler
        self.features = {}
        self.timer = QTimer()
        self.calc_id = "calculator"
        self.calc = None
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
        self.addItem(self.calc_id)
        for k in self.features.keys():
            self.addItem(k)
        return

    @pyqtSlot(QListWidgetItem)
    def item_clicked(self, item):

        # WHERE does this enumerated list of detector types come from?

        if item.text() == self.calc_id:
            if self.calc is None:
                print('create calculator widget')
                self.calc = Calculator(self.comm_handler)
            self.calc.show()
        elif self.features[item.text()] == DataTypes.Image:
            self._spawn_window('AreaDetector', item.text())
            print('create area detector window for:', item.text())
            # cpo/weninc test of "request" pattern for AMI
            #self.comm_handler.sock.send_string("reqimages:%s:3"%item.text())
            #reply = self.comm_handler.sock.recv_string()

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


class AmiGui(QWidget):
    def __init__(self, queue, addr, ami_save, parent=None):
        super(__class__, self).__init__(parent)
        self.setWindowTitle("AMI Client")
        self.comm_handler = CommunicationHandler(addr)

        self.setupLabel = QLabel('Setup:', self)
        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save)
        self.load_button = QPushButton('Load', self)
        self.load_button.clicked.connect(self.load)
        self.clear_button = QPushButton('Clear', self)
        self.clear_button.clicked.connect(self.clear)
        self.reset_button = QPushButton('Reset Plots', self)
        self.reset_button.clicked.connect(self.reset)
        self.dataLabel = QLabel('Data:', self)
        self.amilist = DetectorList(queue, self.comm_handler)
        if ami_save is not None:
            self.amilist.load(ami_save)

        self.setup = QWidget(self)
        self.setup_layout = QHBoxLayout(self.setup)
        self.setup_layout.addWidget(self.save_button)
        self.setup_layout.addWidget(self.load_button)
        self.setup_layout.addWidget(self.clear_button)

        self.ami_layout = QVBoxLayout(self)
        self.ami_layout.addWidget(self.setupLabel)
        self.ami_layout.addWidget(self.setup)
        self.ami_layout.addWidget(self.dataLabel)
        self.ami_layout.addWidget(self.reset_button)
        self.ami_layout.addWidget(self.amilist)

    @pyqtSlot()
    def load(self):
        load_file = QFileDialog.getOpenFileName(self, "Open file", "", "AMI Autosave files (*.ami);;All Files (*)")
        if load_file[0]:
            try:
                with open(load_file[0], 'r') as cnf:
                    self.amilist.load(json.load(cnf))
            except OSError as os_exp:
                print("ami-client: problem opening saved graph configuration file:", os_exp)
            except json.decoder.JSONDecodeError as json_exp:
                print("ami-client: problem parsing saved graph configuration file (%s):"%load_file[0], json_exp)

    @pyqtSlot()
    def save(self):
        save_file = QFileDialog.getSaveFileName(self,"Save file", "autosave.ami", "AMI Autosave files (*.ami);;All Files (*)")
        if save_file[0]:
            print("ami-client: saving graph configuration to file (%s)"%save_file[0])
            try:
                with open(save_file[0], 'w') as cnf:
                    json.dump(self.comm_handler.graph, cnf, indent=4, sort_keys=True)
            except OSError as os_exp:
                print("ami-client: problem opening saved graph configuration file:", os_exp)

    @pyqtSlot()
    def reset(self):
        if not self.comm_handler.reset():
            print("ami-client: unable to reset feature store of the manager!")

    @pyqtSlot()
    def clear(self):
        if not self.comm_handler.clear():
            print("ami-client: unable to clear the graph configuration of the manager!")


def run_main_window(queue, addr, ami_save):
    app = QApplication(sys.argv)
    gui = AmiGui(queue, addr, ami_save)
    gui.show()

    # wait for the qt app to exit
    retval = app.exec_()

    # send exit signal to master process
    queue.put(("exit", None))

    return retval


def run_widget(queue, window_type, topic, addr):

    ctx = zmq.Context()
    app = QApplication(sys.argv)
    win = QMainWindow()

    if window_type == 'AreaDetector':
        widget = AreaDetWidget(topic, addr, win)

    elif window_type == 'WaveformDetector':
        widget = WaveformWidget(topic, addr, win)

    elif window_type == 'ScalarDetector':
        widget = ScalarWidget(topic, addr, win)

    else:
        raise ValueError('%s not valid window_type' % window_type)

    win.setCentralWidget(widget)
    win.setWindowTitle(topic)
    win.show()

    return app.exec_()


def run_client(addr, load):
    saved_cfg = None
    if load is not None:
        try:
            with open(load, 'r') as cnf:
                saved_cfg = json.load(cnf)
        except OSError as os_exp:
            print("ami-client: problem opening saved graph configuration file:", os_exp)
            return 1
        except json.decoder.JSONDecodeError as json_exp:
            print("ami-client: problem parsing saved graph configuration file (%s):"%load, json_exp)
            return 1


    queue = mp.Queue()
    list_proc = mp.Process(target=run_main_window, args=(queue, addr, saved_cfg))
    list_proc.start()
    widget_procs = []

    while True:
        window_type, topic = queue.get()
        if window_type == 'exit':
            print("received exit signal - exiting!")
            break;
        print("opening new widget:", window_type, topic)
        proc = mp.Process(target=run_widget, args=(queue, window_type, topic, addr))
        proc.start()
        widget_procs.append(proc)


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
    addr = "tcp://%s:%d"%(args.host, args.port)

    try:
        return run_client(addr, args.load)
    except KeyboardInterrupt:
        print("Client killed by user...")
        return 0


if __name__ == '__main__':
    sys.exit(main())
