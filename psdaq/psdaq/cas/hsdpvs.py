#!/usr/bin/env python
"""
This implements a PVAccess server that defines the PVs for a HPS server diagnostic bus for conversion to BLD
Any change to the diagnostic bus description (field names, mask) results in an updated payload structure
"""
import os
import sys
import logging
import time
import threading

from p4p.nt import NTScalar
from p4p.server import Server, StaticProvider
from p4p.server.thread import SharedPV
from p4p import Value, Type

from psdaq.hsd.pvdef import *

#  My alarm implementation
import json
from enum import Enum

class AlarmSevr(Enum):
    NONE = 0
    MINOR = 1
    MAJOR = 2
    INVALID = 3

class AlarmStatus(Enum):
    NONE = 0
    READ = 1
    WRITE = 2
    HIGH  = 3
    HIHI  = 4
    LOW   = 5
    LOLO  = 6
    STATE = 7
    CHANGE_OF_STATE = 8
    COMM  = 9
    TIMEOUT = 10

#import pyrogue as pr

logger = logging.getLogger(__name__)
alarms = {}

# pv_fieldname, pv_fieldtype, default_value, mmap_address, bit_size, bit_shift

def pvTypes(nt):
    result = []
    for x,y in nt.items():
        result.append((x,y[0]))
    return result

def pvValues(nt):
    result = []
    for x,y in nt.items():
        result.append((x,y[1]))
    return result

def MySharedPV(nt,cb=None):
    return SharedPV(initial=Value(Type(pvTypes(nt)),pvValues(nt)),
                    handler=DefaultPVHandler(cb))

class DefaultPVHandler(object):
    type = None

    def __init__(self,callback=None):
        self.callback = callback

    def put(self, pv, op):
        postedval = op.value()
#        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time()), 1.0)
        pv.post(postedval)
        op.done()
        if self.callback is not None:
            self.callback(postedval)

class PVHandlerAlarm(object):
    type = None

    def __init__(self,name,callback=None):
        self.name     = name
        self.callback = callback

    def put(self, pv, op):
        postedval = op.value()
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
        limits = alarms[self.name]
        if postedval['value'] < limits[0]:
            postedval['alarm'] = {'severity': AlarmSevr.MAJOR.value,
                                  'status'  : AlarmStatus.LOLO.value}
        elif postedval['value'] > limits[1]:
            postedval['alarm'] = {'severity': AlarmSevr.MAJOR.value,
                                  'status'  : AlarmStatus.HIHI.value}
        else:
            postedval['alarm'] = {'severity': AlarmSevr.NONE.value,
                                  'status'  : AlarmStatus.NONE.value}

        pv.post(postedval)
        op.done()
        if self.callback is not None:
            self.callback(postedval)

class ChipServer(object):
    def __init__(self, provider, prefix, start):
        self.provider = provider
        self.prefix = prefix

        def setLimits(name):
            global alarms
            if name not in alarms:
                print(f'Limits for {name} not found.  Defaulting to NO ALARMS')
                alarms[name] = [-10000000000,10000000000]
        
        #  Make configuration one PV access for each readout channel
        if start:
            daqConfig['enable'] = ('i',1)
        self.daqConfig      = MySharedPV(daqConfig,self.updateDaqConfig)
        self.ready          = SharedPV(initial=NTScalar('I').wrap({'value' : 0}),
                                       handler=DefaultPVHandler())

        self.provider.add(prefix+':CONFIG',self.daqConfig)
        self.provider.add(prefix+':READY' ,self.ready)

        self.pgpConfig      = MySharedPV(pgpConfig)
        self.provider.add(prefix+':PGPCONFIG',self.pgpConfig)

        #  Monitoring
        self.fwBuild     = SharedPV(initial=NTScalar('s').wrap({'value':''}),
                                    handler=DefaultPVHandler())
        self.fwVersion   = SharedPV(initial=NTScalar('I').wrap({'value':0}),
                                    handler=DefaultPVHandler())
        self.pAddr       = SharedPV(initial=NTScalar('s').wrap({'value':''}),
                                    handler=DefaultPVHandler())
        self.pAddr_u     = SharedPV(initial=NTScalar('I').wrap({'value':0}),
                                    handler=DefaultPVHandler())
        self.pLink       = SharedPV(initial=NTScalar('I').wrap({'value':0}),
                                    handler=DefaultPVHandler())
        self.keepRows    = SharedPV(initial=NTScalar('I').wrap({'value':3}),
                                    handler=DefaultPVHandler())
        setLimits(prefix+':FEXOOR')
        self.fexOor      = SharedPV(initial=NTScalar('I',valueAlarm=True).wrap({'value':0}),
                                    handler=PVHandlerAlarm(prefix+':FEXOOR'))

        self.monTiming   = MySharedPV(monTiming)
        self.monPgp      = MySharedPV(monPgp)
        self.monRawBuf   = MySharedPV(monBuf)
        self.monFexBuf   = MySharedPV(monBuf)
        self.monRawDet   = MySharedPV(monBufDetail)
        self.monFexDet   = MySharedPV(monBufDetail)
        self.monFlow     = MySharedPV(monFlow)
        self.monEnv      = MySharedPV(monEnv)
        self.monAdc      = MySharedPV(monAdc)
        self.monJesd     = MySharedPV(monJesd)
        self.monJesdTtl  = MySharedPV(monJesdTtl)

        self.provider.add(prefix+':FWBUILD'   ,self.fwBuild)
        self.provider.add(prefix+':FWVERSION' ,self.fwVersion)
        self.provider.add(prefix+':PADDR'     ,self.pAddr)
        self.provider.add(prefix+':PADDR_U'   ,self.pAddr_u)
        self.provider.add(prefix+':PLINK'     ,self.pLink)
        self.provider.add(prefix+':KEEPROWS'  ,self.keepRows)
        self.provider.add(prefix+':FEXOOR'    ,self.fexOor)
        self.provider.add(prefix+':MONTIMING' ,self.monTiming)
        self.provider.add(prefix+':MONPGP'    ,self.monPgp)
        self.provider.add(prefix+':MONRAWBUF' ,self.monRawBuf)
        self.provider.add(prefix+':MONFEXBUF' ,self.monFexBuf)
        self.provider.add(prefix+':MONRAWDET' ,self.monRawDet)
        self.provider.add(prefix+':MONFEXDET' ,self.monFexDet)
        self.provider.add(prefix+':MONFLOW'   ,self.monFlow)
        self.provider.add(prefix+':MONENV'    ,self.monEnv)
        self.provider.add(prefix+':MONADC'    ,self.monAdc)
        self.provider.add(prefix+':MONJESD'   ,self.monJesd)
        self.provider.add(prefix+':MONJESDTTL',self.monJesdTtl)

        #  Expert functions
        self.daqReset = MySharedPV(daqReset,self.updateDaqReset)
        self.provider.add(prefix+':RESET',self.daqReset)

    def updateDaqConfig(self,value):
        pass

    def updateDaqReset(self,value):
        pass

class PVAServer(object):
    def __init__(self, provider_name, prefix, start):
        self.provider = StaticProvider(provider_name)
        for p in prefix:
            self.chip     = ChipServer(self.provider, p, start)

    def forever(self):
        Server.forever(providers=[self.provider])


def update_limits(alarmFile):
    global alarms
    try:
        last = os.stat(alarmFile)
        alarms |= json.load(open(alarmFile,'r'))
        print(f'Alarms updated')
        print(alarms)
    except:
        print(f'Error opening alarms file {alarmFile}.')
        
    while alarms:
        time.sleep(5)
        try:
            curr = os.stat(alarmFile)
            if curr.st_mtime > last.st_mtime:
                print(f'Alarms file {alarmFile} updated.')
                last = curr
                alarms |= json.load(open(alarmFile,'r'))
                print(f'Alarms updated')
                print(alarms)
        except:
            print(f'Error opening alarms file {alarmFile}.')


import argparse

def main():
    global alarms
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for High Speed Digitizer')

    parser.add_argument('-P', required=True, help='DAQ:LAB2:HSD:DEV06_3E', nargs='+', metavar='PREFIX')
    parser.add_argument('-A', required=False, default='', help='Alarm limits input file', metavar='FILE')
    parser.add_argument('-s', '--start', action='store_true', help='start acq')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    server = PVAServer(__name__, args.P, args.start)

    t = None
    if args.A:
        t = threading.Thread(target=update_limits, args=(args.A,))
        t.start()

    try:
        # process PVA transactions
        server.forever()
    except KeyboardInterrupt:
        print('\nInterrupted')

    if t:
        alarms = None
        t.join()

if __name__ == '__main__':
    main()
