import sys
import zmq
from datetime import datetime, timezone
from PyQt5 import QtCore, QtGui, QtWidgets

PORT_BASE = 29980
POSIX_TIME_AT_EPICS_EPOCH = 631152000

def timestampStr():
    current = datetime.now(timezone.utc)
    nsec = 1000 * current.microsecond
    sec = int(current.timestamp()) - POSIX_TIME_AT_EPICS_EPOCH
    return '%010d-%09d' % (sec, nsec)


def create_msg(key, msg_id=None, sender_id=None, body={}):
    if msg_id is None:
        msg_id = timestampStr()
        msg = {'header': {
               'key': key,
               'msg_id': msg_id,
               'sender_id': sender_id},
           'body': body}
    return msg

def rep_port(platform):
    return PORT_BASE + platform + 20


class CollectionWidget(QtWidgets.QWidget):
    def __init__(self, partition, parent=None):
        super().__init__(parent)
        self.context = zmq.Context(1)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect('tcp://drp-tst-acc06:%d' % rep_port(partition))

        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel('Collection') , 0, 0, 1, 3)

        l = QtWidgets.QHBoxLayout()
        button = QtWidgets.QPushButton('Auto connect')
        button.clicked.connect(self.auto_connect)
        l.addWidget(button)

        button = QtWidgets.QPushButton('Reset')
        button.clicked.connect(self.reset)
        l.addWidget(button)
        layout.addLayout(l, 1, 0, 1, 3)

        self.label = QtWidgets.QLabel()
        layout.addWidget(self.label, 2, 0, 1, 3)

        self.listWidgets = {}
        for i, group in enumerate(['drp', 'teb', 'meb']):
            layout.addWidget(QtWidgets.QLabel(group), 3, i)
            w = QtWidgets.QListWidget()
            layout.addWidget(w, 4, i)
            self.listWidgets[group] = w
        self.setLayout(layout)
        self.setMaximumWidth(300)

    def auto_connect(self):
        self.label.clear()
        for w in self.listWidgets.values():
            w.clear()
        for cmd in ['plat', 'alloc', 'connect']:
            self.socket.send_json(create_msg(cmd))
            response = self.socket.recv_json()
            print(response)
            if 'error' in response['body']:
                self.label.setText(response['body']['error'])
                return
        self.get_state()


    def reset(self):
        self.label.clear()
        for w in self.listWidgets.values():
            w.clear()
        self.socket.send_json(create_msg('reset'))
        print(self.socket.recv_json())


    def get_state(self):
        msg = create_msg('getstate')
        self.socket.send_json(msg)
        reply = self.socket.recv_json()
        for group in reply['body']:
            if group not in self.listWidgets:
                print('unknown group:', group)
                continue
            w = self.listWidgets[group]
            w.clear()
            for k, v in reply['body'][group].items():
                host = v['proc_info']['host']
                QtWidgets.QListWidgetItem(host, w)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])                      
    widget = CollectionWidget()
    widget.show()
    sys.exit(app.exec_())
