#--------------------
"""
:py:class:`QWZMQListener` - widget for integration of zmq messages in qt event loop
===================================================================================
Test::
    python test_QWZMQListener.py # submits messages
    python QWZMQListener.py      # receives messages

Usage::
    # Import
    from psdaq.control_gui.QWZMQListener import QWZMQListener

See:
    - test below
    - :py:class:`QWZMQListener`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-02-07 by Mikhail Dubrovin
"""
#----------

import zmq
from PyQt5.QtCore import QSocketNotifier
from PyQt5.QtWidgets import QWidget

class QWZMQListener(QWidget):
    def __init__(self, **kwargs):
        QWidget.__init__(self)

        _on_poll     = kwargs.get('on_poll', self.on_zmq_poll)
        _port        = kwargs.get('port', 'tcp://localhost:5556')
        _topicfilter = kwargs.get('topicfilter', b'10001')

        self.init_connect_zmq(_on_poll, _port, _topicfilter)


    def init_connect_zmq(self, on_poll, port, topicfilter):
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.SUB)
        self._zmq_socket.connect(port)
        self._zmq_socket.setsockopt(zmq.SUBSCRIBE, topicfilter) # topicfilter = b"10001"

        self._notifier = QSocketNotifier(self._zmq_socket.getsockopt(zmq.FD), QSocketNotifier.Read, self)
        self._notifier.activated.connect(on_poll)


    def on_zmq_poll(self):
        """Needs to be re-implemented to do real work with messages from zmq.
        """
        self._notifier.setEnabled(False)
        flags = self._zmq_socket.getsockopt(zmq.EVENTS)

        #print('in on_read_msg flags', flags)
        if flags & zmq.POLLIN :
            #print("flag zmq.POLLOUT")
            msg = self._zmq_socket.recv_multipart()
            print('flag zmq.POLLIN %d received msg: %s' % (flags,msg))
            self.setWindowTitle(str(msg))
        elif flags & zmq.POLLOUT :
            print("flag zmq.POLLOUT", flags)
        elif flags & zmq.POLLERR :
            print("flag zmq.POLLERR", flags)
        else : print("flag is UNKNOWN", flags)

        self._notifier.setEnabled(True)
        _ = self._zmq_socket.getsockopt(zmq.EVENTS) # WITHOUT THIS LINE IT WOULD NOT CALL on_read_msg AGAIN!

#----------

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    win = QWZMQListener()
    win.show()
    app.exec_()

#----------
