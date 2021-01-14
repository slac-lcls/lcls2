"""
Class :py:class:`CMQThreadClient` -  sub-class of QThread
==========================================================

Usage ::
    from psana.graphqt.CMQThreadClient import CMQThreadClient

    t1 = CMQThreadClient()
    t1.connect_client_is_ready_to(receive_client_is_ready)
    t1.start()

    def receive_client_is_ready():
        client = t1.client()
        stat = t1.quit()

See:
    - :class:`CMWMain`
    - :class:`CMConfigParameters`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2018-05-30 by Mikhail Dubrovin
"""

from psana.graphqt.CMDBUtils import connect_client, database_names
from PyQt5.QtCore import QThread, pyqtSignal

#---

class CMQThreadClient(QThread):
    client_is_ready = pyqtSignal()

    def __init__(self, parent=None, host=None, port=None):
        """Connects to MongoDB client
        """
        QThread.__init__(self, parent)
        self._client = None
        self._host = host
        self._port = port


    def run(self):
        """Launched on start()
        """
        #logger.debug('In CMQThreadClient.run - connect client')
        self._client = connect_client(self._host, self._port)

        if self._client is None:
            #logger.warning("Can't connect to server")
            #print("Can't connect to server")
            pass
        else:
            #logger.debug('Connection to server is ready')
            #print('Connection to server is ready')
            pass
            
        self.client_is_ready.emit()


    def connect_client_is_ready_to(self, recip):
        self.client_is_ready.connect(recip)


    def disconnect_client_is_ready_from(self, recip):
        self.client_is_ready.disconnect(recip)


    def receive_client_is_ready(self):
        if self._client is None:
            print("Can't connect to server")

        dbnames = database_names(self._client)
        #logger.debug('Signal "client_is_ready" received, dbnames: %s' % str(dbnames))
        print('Signal "client_is_ready" received, dbnames: %s' % str(dbnames))


    def client(self):
        return self._client


    def is_running(self):
        return self.isRunning()


    def is_finished(self):
        return self.isFinished()



if __name__ == "__main__":

    import sys

    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s %(name)s: %(message)s', level=logging.DEBUG)

    def on_but_play():
        logger.debug('XXX: on_but_play')

    def on_but_exit():
        logger.debug('XXX: on_but_exit')
        sys.exit()

    from PyQt5.QtWidgets import QApplication, QPushButton, QHBoxLayout, QWidget

    app = QApplication(sys.argv)
    t1 = CMQThreadClient(host='psanaphi103', port=27017)
    t1.connect_client_is_ready_to(t1.receive_client_is_ready)
    t1.start()
    #stat = t1.quit()

    t2 = CMQThreadClient(host='psanaphi105', port=27017)
    t2.connect_client_is_ready_to(t2.receive_client_is_ready)
    t2.start()
    #stat = t2.quit()

    b1 = QPushButton('Play')
    b2 = QPushButton('Exit')
    hbox = QHBoxLayout()
    hbox.addWidget(b1)
    hbox.addWidget(b2)
    w = QWidget()
    w.setLayout(hbox)
    w.show()
    b1.clicked.connect(on_but_play)
    b2.clicked.connect(on_but_exit)
    stat = app.exec_()

# EOF
