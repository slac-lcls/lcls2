#!/usr/bin/env python

import sys
import pyrogue as pr
import logging
from hpsbld._HpsRoot import HpsRoot

from p4p.nt import NTScalar
from p4p.server import Server, StaticProvider
from p4p.server.thread import SharedPV
from p4p.client.thread import Context
from p4p import Value, Type
import threading

import time
from datetime import datetime

pv = None
bldName = None

class DefaultPVHandler(object):

    def __init__(self, parent):
        self.parent = parent

    def put(self, pv, op):
        postedval = op.value()
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time()), 1.0)
        pv.post(postedval)
        op.done()
        self.parent.update()

#
#  Host the PVs used by DAQ control
#
class PVCtrls(threading.Thread):
    def __init__(self, prefix, app):
        threading.Thread.__init__(self,daemon=True)
        self.prefix = prefix+':'
        self.app    = app

    def run(self):
        self.provider = StaticProvider(__name__)

        self.fieldNames = SharedPV(initial=NTScalar('as').wrap
                                   ({'value' : ['pid%02x'%i for i in range(31)]}),
                                   handler=DefaultPVHandler(self))

        # 'i' (integer) or 'f' (float)
        self.fieldTypes = SharedPV(initial=NTScalar('aB').wrap({'value' : [ord('i')]*31}),
                                   handler=DefaultPVHandler(self))

        self.fieldMask  = SharedPV(initial=NTScalar('I').wrap({'value' : 0x8000}),
                                   handler=DefaultPVHandler(self))

        self.payload    = SharedPV(initial=Value(Type([]),{}), 
                                   handler=DefaultPVHandler(self))

        print('Hosting {:}HPS:FIELDMASK'.format(self.prefix))
        self.provider.add(self.prefix+'HPS:FIELDNAMES',self.fieldNames)
        self.provider.add(self.prefix+'HPS:FIELDTYPES',self.fieldTypes)
        self.provider.add(self.prefix+'HPS:FIELDMASK' ,self.fieldMask)
        self.provider.add(self.prefix+'PAYLOAD'   ,self.payload)
        self.update()

        try:
            Server.forever(providers=[self.provider])
        except:
            print('Server exited')

    def update(self):
        self.app.Activate.set(0)

        mask  = self.fieldMask .current().get('value')
        names = self.fieldNames.current().get('value')
        types = self.fieldTypes.current().get('value')
        oid   = self.payload   .current().getID()
        nid   = str(mask)

        print('PVCtrls.update mask[{:x}] oid[{:}]'.format(mask,oid))

        if nid==oid:
            nid += 'a'
        ntypes  = []
        nvalues = {}
        ntypes.append( ('valid', 'i') )
        nvalues[ 'valid' ] = 0
        mmask = mask
        for i in range(31):
            if mmask&1:
                ntypes.append( (names[i], chr(types[i])) )
                nvalues[ names[i] ] = 0
            mmask >>= 1

        pvname = self.prefix+'PAYLOAD'
        self.provider.remove(pvname)
        self.payload = SharedPV(initial=Value(Type(ntypes,id=nid),nvalues), handler=DefaultPVHandler(self))
        print('Payload struct ID %s'%self.payload.current().getID())
        self.provider.add(pvname,self.payload)

        if mask:
            self.app.ChannelMask.set(mask)
            self.app.Activate.set(1)


def hps_init(name, ipAddr, pktSize):
    global pv
    global bldName

    bldName = name

    myargs = { 'ipAddr'      : ipAddr,
               'pollEn'      : False,
               'initRead'    : True,
               'serverPort'  : None }

    root = HpsRoot(**myargs)
    root.__enter__()

    root.Top.AxiSy56040.OutputConfig[1].set(2)

    cntl = root.Top.BldControl
    cntl.MaxSize.set(pktSize)

    pv = PVCtrls(bldName, cntl)
    pv.start()

    return root
    
def hps_connect(root):

    ctxt = Context('pva')
    d = {}
    d['addr'] = ctxt.get(bldName+':ADDR')
    d['port'] = ctxt.get(bldName+':PORT')
    print('hps_connect {:}'.format(d))
    ctxt.close()

    return d
