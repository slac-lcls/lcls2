#!/usr/bin/env python

#
#  Launch this prior to Configure.  It will close after Unconfigure
#
from p4p.server import Server, StaticProvider
from p4p.server.thread import SharedPV
from p4p.nt import NTScalar
import threading
import time
import datetime

class PVHandler(object):

    def __init__(self,cb):
        self._cb = cb

    def put(self, pv, op):
        postedval = op.value()
        print('PVHandler put {:} {:}'.format(postedval['value'],self._cb))
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
        pv.post(postedval)
        self._cb(pv,postedval['value'])
        op.done()

#
#  Host the PVs used by DAQ control
#
class PVCtrls(threading.Thread):
    def __init__(self, name, app):
        threading.Thread.__init__(self,daemon=True)
        self._name = name
        self._app  = app.XpmMini
        # initialize timestamp
        tnow = datetime.datetime.now()
        t0   = datetime.datetime(1990,1,1)  # epics epoch
        ts = int((tnow-t0).total_seconds())<<32
        app.TPGMiniCore.TStampWr.set(ts)

    def run(self):
        self.provider = StaticProvider(__name__)

        self._pv = []
        self._msgHeader = 0

        def addPV(label,cmd):
            pv = SharedPV(initial=NTScalar('I').wrap(0), 
                          handler=PVHandler(cmd))
            name = self._name+':'+label
            print('Registering {:}'.format(name))
            self.provider.add(name,pv)
            self._pv.append(pv)

        addPV('GroupL0Reset'    , self.l0Reset)
        addPV('GroupL0Enable'   , self.l0Enable)
        addPV('GroupL0Disable'  , self.l0Disable)
        addPV('GroupMsgInsert'  , self.msgInsert)
        addPV('PART:0:Master'   , self.master)
        addPV('PART:0:MsgHeader', self.msgHeader)

        with Server(providers=[self.provider]):
            while True:
                time.sleep(1)

    def l0Reset(self, pv, val):
        self._app.Config_L0Select_Reset.set(1)
        self._app.Config_L0Select_Reset.set(0)

    def l0Enable(self, pv, val):
        self._app.Config_L0Select_Enabled.set(True)

    def l0Disable(self, pv, val):
        self._app.Config_L0Select_Enabled.set(False)

    def msgInsert(self, pv, val):
        if val>0:
            self._app.PartitionMessage.set(0x8000 | self._msgHeader)

    def msgHeader(self, pv, val):
        self._msgHeader = val

    def master(self, pv, val):
        self._app.HwEnable.set(True)

