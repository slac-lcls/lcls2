from p4p.server.thread import SharedPV
from p4p.nt import NTScalar
from p4p.nt import NTTable
import psdaq.pyxpm.autosave as autosave
import time
import logging
from enum import Enum

provider = None

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

def setVerbose(v):
    pass

def setProvider(v):
    global provider
    provider = v

def toTable(t):
    table = []
    for v in t.items():
        table.append((v[0],v[1][0][1:]))
        n = len(v[1][1])
    return table,n

def toDict(t):
    d = {}
    for v in t.items():
        d[v[0]] = v[1][1]
    return d

def toDictList(t,n):
    l = []
    for i in range(n):
        d = {}
        for v in t.items():
            d[v[0]] = v[1][1][i]
        l.append(d)
    return l

#  Translate from NTScalar type to XTC type
_ctype = {'?':'UINT8',
          's':'CHARSTR',
          'b':'INT8',
          'B':'UINT8',
          'h':'INT16',
          'H':'UINT16',
          'i':'INT32',
          'I':'UINT32',
          'l':'INT64',
          'L':'UINT64',
          'f':'FLOAT',
          'd':'DOUBLE'}

def addPV(name,ctype,init=0,archive=False,valueAlarm=False):
    if archive:
        if len(ctype)==2 and ctype[0]=='a':
            xtype = _ctype[ctype[1]]
        else:
            xtype = _ctype[ctype]
        handler = DefaultPVHandler(name,xtype)
    else:
        handler = DefaultPVHandler()
    pv = SharedPV(initial=NTScalar(ctype,valueAlarm=valueAlarm).wrap(init), handler=handler)
    provider.add(name, pv)
    return pv

def addPVC(name,ctype,init,cmd):
    pv = SharedPV(initial=NTScalar(ctype).wrap(init), 
                  handler=PVHandler(cmd))
    provider.add(name,pv)
    return pv

def addPVT(name,t):
    table,n = toTable(t)
    init    = toDictList(t,n)
    pv = SharedPV(initial=NTTable(table).wrap(init),
                  handler=DefaultPVHandler())
    provider.add(name,pv)
    return pv

def pvUpdate(pv, val):
    value = pv.current()
    value['value'] = val
    value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
    pv.post(value)
    pv._handler.archive(val)

class DefaultPVHandler(object):

    def __init__(self, archive=None, ctype='UINT32'):
        self._archive = archive
        self._ctype   = ctype

    def put(self, pv, op):
        postedval = op.value()
        logging.debug('DefaultPVHandler.put ',pv,postedval['value'])
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
        pv.post(postedval)
        op.done()
        self.archive(postedval['value'])

    def archive(self, val):
        if self._archive:
            autosave.add(self._archive,val,self._ctype)
            
class PVHandler(object):

    def __init__(self, cb, archive=None, ctype='UINT32'):
        self._cb = cb
        self._archive = archive
        self._ctype   = ctype

    def put(self, pv, op):
        postedval = op.value()
        logging.debug('PVHandler.put ',postedval['value'],self._cb)
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
        pv.post(postedval)
        self._cb(pv,postedval['value'])
        op.done()
        self.archive(postedval['value'])

    def archive(self, val):
        if self._archive:
            autosave.add(self._archive,val,self._ctype)
            
