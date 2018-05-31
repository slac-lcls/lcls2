"""
Class :py:class:`CMQThreadWorker` -  sub-class of QThread
==========================================================

Usage ::
    from psana.graphqt.CMQThreadWorker import CMQThreadWorker

See:
    - :class:`CMWMain`
    - :class:`CMConfigParameters`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2018-05-29 by Mikhail Dubrovin
"""
#------------------------------

import logging
logger = logging.getLogger(__name__)

#------------------------------

import sys
import os
import random
#from time import sleep
from psana.graphqt.CMConfigParameters import cp

from PyQt5.QtCore import QThread #, QTimer # Qt, QPoint

#------------------------------

class CMQThreadWorker(QThread) :

    def __init__ (self, parent=None, dt_msec=5) :
        """
           uses/updates cp.list_of_sources
        """
        QThread.__init__(self, parent)        

        self.dt_msec = dt_msec
        self.counter = 0

        #self.timer = QTimer()
        ###self.connect(self.timer, QtCore.SIGNAL('timeout()'), self.on_timeout)
        #self.timer.timeout().connect(self.on_timeout)
        #self.timer.start(self.dt_msec)
        #self.timer.stop()


#    def on_timeout(self) :
#        """Slot for signal on_timeout
#        """
#        self.counter += 1
#        print 'XXX:CMQThreadWorker %d' % (counter)
#        self.timer.start(self.dt_msec)
        #if  cp.flag_nevents_collected : 
        #    cp.flag_nevents_collected = False
        #    self.update_presenter()


#    def set_request_find_sources(self) :
#        self.cp.list_of_sources = None


#    def check_flags(self) :
#        if self.cp.list_of_sources is None : 
#           t0_sec = time()
#           self.cp.list_of_sources = psu.list_of_sources()
           #msg = 'XXX %s.%s consumed time (sec) = %.3f' % (self._name, sys._getframe().f_code.co_name, time()-t0_sec)
           #print msg


    def run(self) :
        while True :
            self.counter += 1
            logger.debug('XXX:CMQThreadWorker %d' % self.counter)
            #self.check_flags()
            #self.emit_check_status_signal()
            self.msleep(self.dt_msec)

#------------------------------

if __name__ == "__main__" :

    import sys
    logging.basicConfig(format='%(levelname)s %(name)s: %(message)s', level=logging.DEBUG)

    def on_but_play() :
        logger.debug('XXX: on_but_play')

    def on_but_exit() :
        #stat = t1.quit()
        logger.debug('XXX: on_but_exit')
        sys.exit()

    from PyQt5.QtWidgets import QApplication, QPushButton, QHBoxLayout, QWidget

    app = QApplication(sys.argv)
    t1 = CMQThreadWorker(parent=None, dt_msec=1000)
    t1.start()

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

#------------------------------
