from p4p.server.thread import SharedPV
from p4p.nt import NTScalar
from p4p.nt import NTTable
import psdaq.pyxpm.autosave as autosave
import time
import logging

provider = None

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

def addPV(name,ctype,init=0):
    pv = SharedPV(initial=NTScalar(ctype).wrap(init), handler=DefaultPVHandler())
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

class DefaultPVHandler(object):

    def __init__(self):
        pass

    def put(self, pv, op):
        postedval = op.value()
        logging.debug('DefaultPVHandler.put ',pv,postedval['value'])
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
        pv.post(postedval)
        op.done()

class PVHandler(object):

    def __init__(self,cb,archive=None):
        self._cb = cb
        self._archive = archive

    def put(self, pv, op):
        postedval = op.value()
        logging.debug('PVHandler.put ',postedval['value'],self._cb)
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
        pv.post(postedval)
        self._cb(pv,postedval['value'])
        op.done()
        if self._archive:
            autosave.add(self._archive,postedval['value'])

