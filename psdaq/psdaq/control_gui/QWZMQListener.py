#--------------------
"""
:py:class:`QWZMQListener` - widget for integration of zmq messages in qt event loop
===================================================================================
Test::
    python psdaq/psdaq/control_gui/test_QWZMQListener.py # launch process submitting zmq messages
    python psdaq/psdaq/control_gui/QWZMQListener.py      # launch process receiving messages

Usage::
    # Import
    from psdaq.control_gui.QWZMQListener import QWZMQListener
    class MyWidget(QWZMQListener):
    # re-implement on_zmq_poll

See:
    - test below
    - :py:class:`QWZMQListener`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-02-07 by Mikhail Dubrovin
"""
#----------

import logging
logger = logging.getLogger(__name__)

import zmq
from PyQt5.QtCore import QSocketNotifier
from PyQt5.QtWidgets import QWidget
from psdaq.control.collection import front_pub_port

class QWZMQListener(QWidget):
    def __init__(self, **kwargs):
        QWidget.__init__(self, parent=None)
        #logger.debug('In QWZMQListener.__init__')

        self.timeout = kwargs.get('timeout', 1000)
        _on_poll     = kwargs.get('on_poll', self.on_zmq_poll)
        _host        = kwargs.get('host', 'localhost')
        _platform    = kwargs.get('platform', 6)
        _topicfilter = kwargs.get('topicfilter', b'') # b'10001'
        _uri         = 'tcp://%s:%d' % (_host, front_pub_port(_platform)) # 'tcp://localhost:30016'
        self.init_connect_zmq(_on_poll, _uri, _topicfilter)


    def init_connect_zmq(self, on_poll, uri, topicfilter):
        logger.debug('QWZMQListener.init_connect_zmq uri=%s' % uri)
        self.zmq_context = zmq.Context(1)
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect(uri)
        self.zmq_socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

        self.zmq_notifier = QSocketNotifier(self.zmq_socket.getsockopt(zmq.FD), QSocketNotifier.Read, self)
        self.zmq_notifier.activated.connect(on_poll)


    def on_zmq_poll(self):
        """Needs to be re-implemented to do real work with messages from zmq.
        """
        self.zmq_notifier.setEnabled(False)
        flags = self.zmq_socket.getsockopt(zmq.EVENTS)
        flag = 'UNKNOWN'
        msg = ''
        if flags & zmq.POLLIN :
            flag = 'POLLIN'
            msg = self.zmq_socket.recv_multipart()
            self.setWindowTitle(str(msg))
        elif flags & zmq.POLLOUT : flag = 'POLLOUT'
        elif flags & zmq.POLLERR : flag = 'POLLERR'
        else : pass
        print("Flag zmq.%s in %d msg: %s" % (flag, flags, msg))

        self.zmq_notifier.setEnabled(True)
        self.kick_zmq() # WITHOUT THIS LINE IT WOULD NOT CALL on_read_msg AGAIN!


    def kick_zmq(self):
        """ WITHOUT THIS LINE IT WOULD NOT CALL on_read_msg AGAIN!
        """
        _flags = self.zmq_socket.getsockopt(zmq.EVENTS)

#----------

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
 
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    win = QWZMQListener()
    win.show()
    app.exec_()

#----------
