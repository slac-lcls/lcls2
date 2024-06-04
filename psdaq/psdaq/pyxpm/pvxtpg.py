import time
from p4p.nt import NTScalar
from p4p.server.thread import SharedPV
from psdaq.pyxpm.pvhandler import *
import logging

provider = None
lock     = None

class PVMmcm(PVHandler):
    def __init__(self, provider, name, idx, dev, chd, cuMode=False):
        logging.info(f'Initializing PVmmcm {idx}')

        self._dev = dev
        self._chd = chd
        self._cuMode = cuMode

        pv = SharedPV(initial=NTScalar('aI').wrap([0*2049]), handler=DefaultPVHandler())
        provider.add(name+':MMCM%d'%idx, pv)
        self._pv = pv
        logging.info(f'Added {name}:MMCM{idx}')

        pv = SharedPV(initial=NTScalar('aI').wrap([0*2049]), handler=DefaultPVHandler())
        provider.add(name+':IMMCM%d'%idx, pv)
        self._ipv = pv
        logging.info(f'Added {name}:IMMCM{idx}')

#        self.update()

        pv = SharedPV(initial=NTScalar('I').wrap(0), handler=PVHandler(self.set))
        provider.add(name+':SetMmcm%d'%idx, pv)
        self._set = pv
        logging.info(f'Added {name}:SetMmcm{idx}')

        pv = SharedPV(initial=NTScalar('I').wrap(0), handler=self)
        provider.add(name+':ResetMmcm%d'%idx, pv)
        self._reset = pv
        logging.info(f'Added {name}:ResetMmcm{idx}')

    def update(self,timev):
        while self._dev.nready.get()==1 and self._cuMode:
            logging.info('Waiting for phase lock: {}',self._dev)
            time.sleep(1)
        logging.info(f'Locked: {self._dev}')

        v = []
        w = []
        r = self._dev
        status = (r.delayValue.get()<< 0) | (r.delayEnd    .get()<<16) | (r.externalLock.get()<<29) | (r.nready      .get()<<30) | (r.internalLock.get()<<31)
        v.append(status)
        w.append(self._dev.sumPeriod.get())
        vlen = self._dev.delayEnd.get()
        for i in range(vlen):
            self._dev.ramAddr.set(i)
            v.append(self._dev.ramData .get())
            w.append(self._dev.ramData1.get())

        pv = self._pv
        value = pv.current()
        value['value'] = v
        value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
        pv.post(value)

        pv = self._ipv
        value = pv.current()
        value['value'] = w
        value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
        pv.post(value)

        self._update_required = 0

    def set(self, pv, val):
        self._dev.delaySet.set(val)

    def put(self, pv, op):
        postedval = op.value()
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
        pv.post(postedval)
        op.done()
        if postedval['value']:
            self._dev.Rescan()
            self.require_update()
            for v in self._chd:
                v.require_update()

    def require_update(self):
        self._update_required = 1

    def dump(self):
        logging.info(self._dev)
        logging.info('delay setting {}'.format(self._dev.delaySet  .get()))
        logging.info('delay value   {}'.format(self._dev.delayValue.get()))
        logging.info('delay end     {}'.format(self._dev.delayEnd  .get()))
        logging.info('external lock {}'.format(self._dev.externalLock.get()))
        logging.info('nready        {}'.format(self._dev.nready.get()))
#        logging.info('status        {:x}'.format(self._dev.status.get()))
        self._dev.dump()

class PVCuPhase(object):

    def __init__(self, provider, name, dev):
        self._dev = dev

        pv = SharedPV(initial=NTScalar('f').wrap(0), handler=DefaultPVHandler())
        provider.add(name+':CuPhase', pv)
        self._pv = pv

    def update(self,timev):
        logging.info('PvCuPhase: {:}/{:} {:.2f}  {:},{:}'.format(self._dev.base.get(),
                                                          self._dev.early.get(),
                                                          self._dev.phase_ns.get(),
                                                          self._dev.gate.get(),
                                                          self._dev.late.get()))
        pv = self._pv
        value = pv.current()
        value['value'] = self._dev.phase_ns.get()
        value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
        pv.post(value)

class PVXTpg(object):

    def __init__(self, p, m, name, xpm, devs, cuMode=False, bypassLock=False):
        global provider
        provider = p
        global lock
        lock     = m

        #  Sanity check first
        #  Check input clock source for the board
        if cuMode:
            if xpm.CuTiming.RxLinkUp.get()==0:
                logging.error("CuTiming not locked")
                #raise RuntimeError("CuTiming not locked")
        else:
            if xpm.UsTiming.RxLinkUp.get()==0:
                logging.error("UsTiming not locked")
                #raise RuntimeError("UsTiming not locked")
            else:
                time.sleep(1)
                xpm.UsTiming.Dump()

        if bypassLock:
            logging.info('Bypassing AMC PLL lock')
            app = xpm.find(name=devs[3][0])[0]
            app.bypassLock.set(1)
            time.sleep(1)

        self._mmcm = []
        for i in range(3,-1,-1):
            mmcmname = devs[i][0]
            self._mmcm.append(PVMmcm(provider, name+':XTPG', i, xpm.find(name=mmcmname)[0], self._mmcm.copy(), cuMode=cuMode))
            if not cuMode:
                break
#        self._cuPhase = PVCuPhase(provider, name+':XTPG', xpm.CuPhase)

    def init(self):
        for v in self._mmcm:
            v.dump()
        timev = divmod(float(time.time_ns()), 1.0e9)
        for v in self._mmcm:
            v.update(timev)
#        self._cuPhase.update(timev)
        
    def update(self):
        timev = divmod(float(time.time_ns()), 1.0e9)
        for v in self._mmcm:
            if v._update_required:
                v.update(timev)
#        self._cuPhase.update(timev)
            
