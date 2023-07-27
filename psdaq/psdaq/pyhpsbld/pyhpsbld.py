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
        self.app.Enable.set(0)

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
            self.app.channelMask.set(mask)
            self.app.Enable.set(1)


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

    #  Reset the board?
#    print('Reloading FPGA')
#    root.Top.AxiVersion.FpgaReload.set(1)
#    print('Are we dead yet?')

    root.Top.AxiSy56040.OutputConfig[1].set(3)
    print(f'Timing crossbar set to {root.Top.AxiSy56040.OutputConfig[1].get()}')
    time.sleep(2)

    print(f'Timing rxLinkUp: {root.Top.UsTiming.RxLinkUp.get()}')

    #  Configure BLD
    bld = root.Top.AmcCarrierBsa.BldControl
    bld.packetSize.set(pktSize)
#    bld.Edef0.rateSel.set(6)
    bld.Edef0.rateSel.set(0)
    bld.Edef0.destSel.set(0x20000)
    bld.Edef0.Enable .set(1)
    bld.Enable.set(1)

    #  Configure BSSS
    bsss = root.Top.AmcCarrierBsa.BsssControl
    bsss.packetSize.set(pktSize)
    bsss.channelMask.set(0x7fffffff)
    bsss.Edef0.rateSel.set(6)
    bsss.Edef0.destSel.set(0x20000)
    bsss.Edef0.Enable .set(1)
    bsss.Enable.set(1)

    #  Configure BSAS
    bsas = root.Top.AmcCarrierBsa.BsasControl.Bsas0
    bsas.channelMask.set(0x7fffffff)
    bsas.channelSevr.set(0x3fffffffffffffff)
    bsas.acquire.RateSel.set(0)
    #bsas.acquire.DestSel.set(0x20000)
    bsas.acquire.DestSel.set(0x0000)
    bsas.acquire.EnableReg.set(1)
    bsas.rowAdvance.RateSel.set(2)
    bsas.rowAdvance.DestSel.set(0x20000)
    bsas.rowAdvance.EnableReg.set(1)
    bsas.tableReset.RateSel.set(3)
    bsas.tableReset.DestSel.set(0x20000)
    bsas.tableReset.EnableReg.set(1)
    bsas.Enable.set(1)

    time.sleep(2)
    print(f'BLD   ClkFreq {bld.diagnClkFreq.get()}  StrobeRate {bld.diagnStrobeRate.get()}  EventRate {bld.eventSel0Rate.get()}')
    print(f'BSSS  ClkFreq {bsss.diagnClkFreq.get()}  StrobeRate {bsss.diagnStrobeRate.get()}  EventRate {bsss.eventSel0Rate.get()}')
    print(f'BSAS  acquire {bsas.acquire.Count.get()}  advance {bsas.rowAdvance.Count.get()}  reset {bsas.tableReset.Count.get()}')

    pv = PVCtrls(bldName, bld)
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
