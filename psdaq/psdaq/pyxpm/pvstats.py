import sys
import time
import traceback
import struct
import logging
from datetime import datetime
from p4p.nt import NTScalar
from p4p.nt import NTTable
from psdaq.pyxpm.pvhandler import *

lock     = None
provider = None
fidPeriod   = 1400e-6/1300.

def updatePv(pv,v,timev):
    if v is not None:
        value = pv.current()
        value['value'] = v
        value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
        pv.post(value)

def updatePvC(pv,v,timev):
    if v is not None:
        value = pv.current()
        if value['value']!=v:
            value['value']=v
            value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
            pv.post(value)
            
class SFPStatus(object):

    NFPLINKS = 14
    sfpStatus  = {'LossOfSignal' : ('ai',[0]*NFPLINKS),
                  'ModuleAbsent' : ('ai',[0]*NFPLINKS),
                  'TxPower'      : ('af',[0]*NFPLINKS),
                  'RxPower'      : ('af',[0]*NFPLINKS)}

    def __init__(self, name, xpm, nLinks=14):
        self._xpm   = xpm
        self._pv    = addPVT(name,self.sfpStatus)
        self._value = toDict(self.sfpStatus)
        self._link  = 0
        self._nlinks= nLinks

        amc = self._xpm.amcs[0]
        mod = amc.SfpSummary.modabs.get()
        logging.info(f'SFPStatus mod {mod:x}')

    def update(self):

        amc = self._xpm.amcs[int(self._link/7)]
        mod = amc.SfpSummary.modabs.get()
        los = amc.SfpSummary.los.get()
        j = self._link % 7
        if los is not None:
            self._value['LossOfSignal'][self._link] = (los>>j)&1
        if mod is not None:
            self._value['ModuleAbsent'][self._link] = (mod>>j)&1
            if ((mod>>j)&1)==0:
                amc.I2cMux.set(j|(1<<3))
                txp,rxp = amc.SfpI2c.get_pwr()
                self._value['TxPower'][self._link] = txp
                self._value['RxPower'][self._link] = rxp

        self._link += 1
        if self._link==self._nlinks:
            self._link = 0
            value = self._pv.current()
            value['value'] = self._value
            value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
            self._pv.post(value)

class QSFPStatus(object):

    NFPLINKS = 8
    qsfpStatus  = {'TxPower'      : ('af',[0]*NFPLINKS),
                   'RxPower'      : ('af',[0]*NFPLINKS)}

    def __init__(self, name, xpm, nLinks=2):
        self._xpm   = xpm
        self._pv    = addPVT(name,self.qsfpStatus)
        self._value = toDict(self.qsfpStatus)
        self._link  = 0
        self._nlinks= nLinks

    def update(self):

        j = self._link % 2
        self._xpm.AxiPcieCore.I2cMux.set((1<<4) if j==0 else (1<<1))
        rxp = self._xpm.AxiPcieCore.QSFP.getRxPwr()
        txp = self._xpm.AxiPcieCore.QSFP.getTxBiasI()
        for lane in range(4):
            self._value['TxPower'][4*self._link+lane] = txp[lane]
            self._value['RxPower'][4*self._link+lane] = rxp[lane]

        self._link += 1
        if self._link==self._nlinks:
            self._link = 0
            value = self._pv.current()
            value['value'] = self._value
            value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
            self._pv.post(value)

class LinkStatus(object):
    def __init__(self, name, app, i):
        self._app = app
        self._idx = i
        self._app.link.set(i)
        self._rxRcv = 0 # self._app.dsLinkRxCnt.get()
        self._rxErr = 0 # self._app.dsLinkStatus.get()&0xffff

        def _addPVI(label):
            return addPV(name+':'+label+'%d'%i,'I')

        self._pv_txReady      = _addPVI('LinkTxReady')
        self._pv_rxReady      = _addPVI('LinkRxReady')
        self._pv_txResetDone  = _addPVI('LinkTxResetDone')
        self._pv_rxResetDone  = _addPVI('LinkRxResetDone')
        self._pv_rxRcv        = _addPVI('LinkRxRcv')
        self._pv_rxErr        = _addPVI('LinkRxErr')
        self._pv_rxIsXpm      = _addPVI('LinkRxIsXpm')
        self._pv_remoteLinkId = _addPVI('RemoteLinkId')

    def handle(self,msg,offset,timev):
        w = struct.unpack_from('<LLL',msg,offset)
        offset += 12
        u = (w[2]<<64) + (w[1]<<32) + w[0]
        updatePv(self._pv_txResetDone,(u>>0)&1,timev)
        updatePv(self._pv_txReady    ,(u>>1)&1,timev)
        updatePv(self._pv_rxResetDone,(u>>2)&1,timev)
        updatePv(self._pv_rxReady    ,(u>>3)&1,timev)

        v = (u>>5)&0xffff
        updatePv(self._pv_rxErr,(v-self._rxErr)&0xffff,timev)
        self._rxErr = v

        v = (u>>21)&0xffffffff
        updatePv(self._pv_rxRcv,(v-self._rxRcv)&0xffffffff,timev)
        self._rxRcv = v

        updatePv(self._pv_rxIsXpm,(u>>53)&1,timev)
        updatePv(self._pv_remoteLinkId, (u>>54)&0xffffffff,timev)
        return offset

class TimingStatus(object):
    def __init__(self, name, device, linkUpdate, fidRate):
        self._name = name
        self._device = device
        self._device.update()
        self._linkUpdate = linkUpdate
        self._rxClkCount      = device.RxClkCount.get()<<4
        self._txClkCount      = device.TxClkCount.get()<<4
        self._rxRstCount      = device.RxRstCount.get()
        self._crcErrCount     = device.CrcErrCount.get()
        self._rxDecErrCount   = device.RxDecErrCount.get()
        self._rxDspErrCount   = device.RxDspErrCount.get()
        self._bypassRstCount  = device.BypassResetCount.get()
        self._bypassDoneCount = device.BypassDoneCount.get()
        self._fidCount        = device.FidCount.get()
        self._sofCount        = device.sofCount.get()
        self._eofCount        = device.eofCount.get()

        self._vLast = 0

        def _addPVF(label,valueAlarm=False):
            return addPV(name+':'+label,'f',valueAlarm=valueAlarm)

        self._pv_rxClkCount  = _addPVF('RxClks')
        self._pv_txClkCount  = _addPVF('TxClks')
        self._pv_rxRstCount  = _addPVF('RxRsts')
        self._pv_crcErrCount = _addPVF('CrcErrs')
        self._pv_rxDecErrs   = _addPVF('RxDecErrs')
        self._pv_rxDspErrs   = _addPVF('RxDspErrs')
        self._pv_bypassRsts  = _addPVF('BypassRsts')
        self._pv_bypassDones = _addPVF('BypassDones')
        self._pv_rxLinkUp    = _addPVF('RxLinkUp',valueAlarm=True)
        self._pv_fids        = _addPVF('FIDs',valueAlarm=True)
        self._fidLimits = [0.95*fidRate,1.05*fidRate]
        self._pv_sofs        = _addPVF('SOFs')
        self._pv_eofs        = _addPVF('EOFs')
        self._pv_rxAlign     = addPV(name+':RxAlign', 'aI', [0]*65)

    def update(self):

        def updatePv(pv,nv,ov,verbose=False,nb=32,limits=None):
            if nv is not None:
                value = pv.current()
                mask = (1<<nb)-1
                result = (nv-ov)&mask if (nv!=ov or (nv&mask)!=((-1)&mask)) else -1.
                value['value'] = result
                value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
                if limits is not None:
                    if result < limits[0]:
                        value['alarm']['severity'] = AlarmSevr.MAJOR.value
                        value['alarm']['status'  ] = AlarmStatus.LOLO.value
                    elif result > limits[1]:
                        value['alarm']['severity'] = AlarmSevr.MAJOR.value
                        value['alarm']['status'  ] = AlarmStatus.HIHI.value
                pv.post(value)
                if type(verbose) is type("") and nv != ov:
                    logging.warning(f'*** {datetime.now()} {self._name+":"+verbose} changed: {ov} -> {nv} @ {timev}')
                return nv
            else:
                return ov

        timev = divmod(float(time.time_ns()), 1.0e9)

        self._device.update()
        self._rxClkCount      = updatePv(self._pv_rxClkCount, self._device.RxClkCount.get()<<4, self._rxClkCount)
#        self._txClkCount      = updatePv(self._pv_txClkCount, self._device.TxClkCount.get()<<4, self._txClkCount)
        self._rxRstCount      = updatePv(self._pv_rxRstCount, self._device.RxRstCount.get(), self._rxRstCount)
        self._crcErrCount     = updatePv(self._pv_crcErrCount, self._device.CrcErrCount.get(), self._crcErrCount)
        self._rxDecErrCount   = updatePv(self._pv_rxDecErrs, self._device.RxDecErrCount.get(), self._rxDecErrCount, "RxDecErrs")
        self._rxDspErrCount   = updatePv(self._pv_rxDspErrs, self._device.RxDspErrCount.get(), self._rxDspErrCount, "RxDspErrs")
        self._bypassRstCount  = updatePv(self._pv_bypassRsts, self._device.BypassResetCount.get(), self._bypassRstCount)
        self._bypassDoneCount = updatePv(self._pv_bypassDones, self._device.BypassDoneCount.get(), self._bypassDoneCount)
        self._fidCount        = updatePv(self._pv_fids, self._device.FidCount.get(), self._fidCount) #, self._fidLimits)
#        self._sofCount        = updatePv(self._pv_sofs, self._device.sofCount.get(), self._sofCount)
#        self._eofCount        = updatePv(self._pv_eofs, self._device.eofCount.get(), self._eofCount)

        oflow = (1<<32)-1
        if (self._rxDecErrCount==oflow or self._rxDspErrCount==oflow):
            self._device.RxCountReset.set(1)
            time.sleep(10.e-6)
            self._device.RxCountReset.set(0)

        v = self._device.RxLinkUp.get()
        if v is not None:
            value = self._pv_rxLinkUp.current()
            value['value'] = v
            if v==0:
                value.alarm.severity = AlarmSevr.MAJOR.value
                value.alarm.status   = AlarmStatus.STATE.value
                value.alarm.message  = 'Input link down'
            else:
                value.alarm.severity = AlarmSevr.NONE.value
                value.alarm.status   = AlarmStatus.NONE.value
                value.alarm.message  = 'Input link up'

            value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
            self._pv_rxLinkUp.post(value)

            if v != self._vLast:
                logging.warning(f'*** {datetime.now()} {self._name}:RxLinkUp changed: {self._vLast} -> {v} @ {timev}')
                self._vLast = v

            #  Link was down but now is up
            if v and self._device.RxDown.get():
                logging.warning(f'*** {datetime.now()} {self._name}:RxDown latched and linkUp')
                self._device.RxDownCTL.set(0)
                if self._linkUpdate:
                    self._linkUpdate()
            
class TimingLock(object):
    def __init__(self, name, dev):
        self._pv_Dump        = addPVC(name+':DUMP', 'I', 0, self.dump)

    def dump(self, pv, val):
        self._dev.dump()
        
class AmcPLLStatus(object):
    def __init__(self, name, app, idx):
        self._idx    = idx
        self._idxreg = app.amc
        self._device = app.amcPLL

        def _addPVI(label):
            return addPV(name+':'+label+'%d'%idx,'I')

        self._pv_lol    = _addPVI('PLL_LOL')
        self._pv_lolCnt = _addPVI('PLL_LOLCNT')
        self._pv_los    = _addPVI('PLL_LOS')
        self._pv_losCnt = _addPVI('PLL_LOSCNT')

        logging.info(f'amcPLL{idx} {app.amcPLL.rstn.get()} {app.amcPLL.bypass.get()}')

    def handle(self, msg, offset, timev):
        w = struct.unpack_from('<B',msg,offset)
        offset += 1
        updatePv(self._pv_los   ,(w[0]>>0)&1, timev)
        updatePv(self._pv_losCnt,(w[0]>>1)&7, timev)
        updatePv(self._pv_lol   ,(w[0]>>4)&1, timev)
        updatePv(self._pv_lolCnt,(w[0]>>5)&7, timev)
        return offset

class CuStatus(object):
    def __init__(self, name, device, phase):
        self._device = device
        self._phase  = phase

        self._pv_timeStamp    = addPV(name+':TimeStamp'   ,'L')
        self._pv_pulseId      = addPV(name+':PulseId'     ,'L')
        self._pv_fiducialIntv = addPV(name+':FiducialIntv','I')
        self._pv_fiducialErr  = addPV(name+':FiducialErr' ,'I')
        self._pv_PhCuToSC     = addPV(name+':CuToSCPhase' ,'f')

    def update(self):

        def updatePv(pv,v):
            if v is not None:
                value = pv.current()
                value['value'] = v
                value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
                pv.post(value)

        timev = divmod(float(time.time_ns()), 1.0e9)
        updatePv(self._pv_timeStamp   , self._device.timeStampSec()         )
        updatePv(self._pv_pulseId     , self._device.pulseId          .get())
        updatePv(self._pv_fiducialIntv, self._device.cuFiducialIntv   .get())
        updatePv(self._pv_PhCuToSC    , self._phase .phase())

        def updatePv(pv,v):
            if v is not None:
                value = pv.current()
                if not (value==v):   # skip redundant updates
                    value['value'] = v
                    value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
                    pv.post(value)

        updatePv(self._pv_fiducialErr , self._device.cuFiducialIntvErr.get())

class NoCuStatus(object):
    def __init__(self, name):

        self._pv_timeStamp    = addPV(name+':TimeStamp'   ,'L')
        self._pv_pulseId      = addPV(name+':PulseId'     ,'L')
        self._pv_fiducialIntv = addPV(name+':FiducialIntv','I')
        self._pv_fiducialErr  = addPV(name+':FiducialErr' ,'I')
        self._pv_PhCuToSC     = addPV(name+':CuToSCPhase' ,'f')

    def update(self):
        pass

class MonClkStatus(object):
    def __init__(self, name, app):
        self._app   = app

        self._pv_bpClk  = addPV(name+':BpClk' ,'f')
        self._pv_fbClk  = addPV(name+':FbClk' ,'f')
        self._pv_recClk = addPV(name+':RecClk','f')

    def handle(self, msg, offset, timev):
        w = struct.unpack_from('<LLLL',msg,offset)
        offset += 16
        updatePv(self._pv_bpClk , w[0]&0xfffffff, timev)
        updatePv(self._pv_fbClk , w[1]&0xfffffff, timev)
        updatePv(self._pv_recClk, w[2]&0xfffffff, timev)
        return offset

class GroupStats(object):
    def __init__(self, name, app, group):
        self._app   = app
        self._group = group
        self._master = 0
        self._timeval = float(time.time_ns())
        self._app.partition.set(group)
        l0Stats        = 0
        self._l0Ena    = self._app.l0EnaCnt(l0Stats)
        self._l0Inh    = self._app.l0InhCnt(l0Stats)
        self._numL0    = self._app.numL0   (l0Stats)
        self._numL0Acc = self._app.numL0Acc(l0Stats)
        self._numL0Inh = self._app.numL0Inh(l0Stats)
        self._linkInhEv = None
        self._linkInhTm = None

        def _addPVF(label):
            return addPV(name+':'+label,'f')

        self._pv_running   = addPV(name+':Running'  ,'b',False)
        self._pv_recording = addPV(name+':Recording','b',False)
        self._pv_runTime   = _addPVF('RunTime')
#        self._pv_msgDelay  = _addPVF('MsgDelay')
        self._pv_l0InpRate = _addPVF('L0InpRate')
        self._pv_l0AccRate = _addPVF('L0AccRate')
        self._pv_l1Rate    = _addPVF('L1Rate')
        self._pv_numL0Inp  = _addPVF('NumL0Inp')
        self._pv_numL0Inh  = _addPVF('NumL0Inh')
        self._pv_numL0Acc  = _addPVF('NumL0Acc')
        self._pv_numL1     = _addPVF('NumL1')
        self._pv_numDlyOF  = _addPVF('NumDlyOF')
        self._pv_deadFrac  = _addPVF('DeadFrac')
        self._pv_deadTime  = _addPVF('DeadTime')

        self._pv_deadFLink = addPV(name+':DeadFLnk','af',[0.]*32)

    def handle(self,msg,offset,timev):
        global fidPeriod
        linkInhEv = {}
        linkInhTm = {}
        for k in range(32):
            linkInhEv[k]=struct.unpack_from('<L',msg,offset)[0]
            offset += 4
            linkInhTm[k]=struct.unpack_from('<L',msg,offset)[0]
            offset += 4

        def bytes2Int(msg,offset):
            b = struct.unpack_from('<BBBBB',msg,offset)
            offset += 5
            w = 0
            for i,v in enumerate(b):
                w += v<<(8*i)
            return (w,offset)

        (l0Ena   ,offset) = bytes2Int(msg,offset)
        (l0Inh   ,offset) = bytes2Int(msg,offset)
        (numL0   ,offset) = bytes2Int(msg,offset)
        (numL0Inh,offset) = bytes2Int(msg,offset)
        (numL0Acc,offset) = bytes2Int(msg,offset)
        numDlyOF = struct.unpack_from('<B',msg,offset)[0]
        offset += 1

        rT = l0Ena*fidPeriod
        updatePv(self._pv_runTime , rT, timev)
#       Does this get() cause problems via multi-threading?
#        updatePv(self._pv_msgDelay, self._app.l0Delay.get(), timev)

        if self._master:
            dL0Ena   = l0Ena    - self._l0Ena
            dL0Inh   = l0Inh    - self._l0Inh
            dt       = dL0Ena*fidPeriod
            dnumL0   = numL0    - self._numL0
            dnumL0Acc= numL0Acc - self._numL0Acc
            dnumL0Inh= numL0Inh - self._numL0Inh
            if dL0Ena:
                l0InpRate = dnumL0/dt
                l0AccRate = dnumL0Acc/dt
                updatePv(self._pv_deadTime, dL0Inh/dL0Ena, timev)
            else:
                l0InpRate = 0
                l0AccRate = 0
            updatePv(self._pv_l0InpRate, l0InpRate, timev)
            updatePv(self._pv_l0AccRate, l0AccRate, timev)
            updatePv(self._pv_numL0Inp, numL0, timev)
            updatePv(self._pv_numL0Inh, numL0Inh, timev)
            updatePv(self._pv_numL0Acc, numL0Acc, timev)
            if dnumL0:
                deadFrac = dnumL0Inh/dnumL0
            else:
                deadFrac = 0
            updatePv(self._pv_deadFrac, deadFrac, timev)
            updatePvC(self._pv_running, dL0Ena>0, timev)

            self._l0Ena   = l0Ena
            self._l0Inh   = l0Inh
            self._numL0   = numL0
            self._numL0Acc= numL0Acc
            self._numL0Inh= numL0Inh
        else:
            updatePvC(self._pv_running, False, timev)

        updatePv(self._pv_numDlyOF, numDlyOF, timev)

        if True:
            if self._linkInhTm:
                den = fidPeriod
                linkInhTmV = []
                for i in range(32):
                    linkInhTmV.append((linkInhTm[i] - self._linkInhTm[i])*den)
            else:
                linkInhTmV = [0 for i in range(32)]

            self._linkInhTm = linkInhTm
            updatePv(self._pv_deadFLink, linkInhTmV, timev)

        return offset

NGROUPS = 8
pattStats  = {'Sum'    : ('ai',[0]*NGROUPS),
              'First'  : ('ai',[0]*NGROUPS),
              'Last'   : ('ai',[0]*NGROUPS),
              'MinIntv': ('ai',[0]*NGROUPS),
              'MaxIntv': ('ai',[0]*NGROUPS)}
NCOINC = NGROUPS*(NGROUPS+1)//2
pattCoinc = {'Coinc' : ('ai',[0]*NCOINC)}

class PatternStats(object):
    def __init__(self, name):
        # statistics
        self._stats_pv    = addPVT(f'{name}:GROUPS',pattStats)
        self._stats_value = toDict(pattStats)

        self._coinc_pv    = addPVT(f'{name}:COINC',pattCoinc)
        self._coinc_value = toDict(pattCoinc)

    def handle(self,msg,offset,timev):

        def bytes22Int(msg,words,offset):
            if words%2:
                raise ValueError
            nb = 5*words//2
            b = struct.unpack_from(f'<{nb}B',msg,offset)
            offset += nb
            a = []
            w = 0
            for i,v in enumerate(b):
                w += v<<(8*(i%5))
                if (i%5)==4:
                    a.append(w&0xfffff)
                    w >>= 20
                    a.append(w&0xfffff)
                    w = 0
                    
            #print(f'inp {words} len {len(a)} {a}')
            return (a,offset)

        (a, offset) = bytes22Int(msg,NGROUPS*5,offset)
        for i in range(NGROUPS):
            self._stats_value['Sum'    ][i] = a[5*i+0]
            self._stats_value['First'  ][i] = a[5*i+1]
            self._stats_value['Last'   ][i] = a[5*i+2]
            self._stats_value['MinIntv'][i] = a[5*i+3]
            self._stats_value['MaxIntv'][i] = a[5*i+4]

        (a, offset) = bytes22Int(msg,NCOINC,offset)
        for i,v in enumerate(a):
            self._coinc_value['Coinc'][i] = v

        updatePv(self._stats_pv, self._stats_value, timev)
        updatePv(self._coinc_pv, self._coinc_value, timev)

        return offset
        
class PVMmcmPhaseLock(object):
    def __init__(self, name, mmcm):
        v = []
        v.append( mmcm.delayValue.get() )
        v.append( mmcm.waveform.get() )
        self.pv   = addPV(name,'ai',v)

class PathTimer(object):
    def __init__(self, name, xpm, group):
        self._pathTimer = getattr(xpm,f'XpmPathTimer_{group}')
        
        self._pv_latched = addPV(f'{name}:PART:{group}:PATH_TIME:Latched','I')
        self._pv_array   = addPV(f'{name}:PART:{group}:PATH_TIME:Array','ai',[0]*14)

        def _addPV(label, handler):
            pv = SharedPV(initial=NTScalar('I').wrap(0), 
                          handler=handler)
            provider.add(label,pv)
            return pv

        self._pv_update  = _addPV(f'{name}:PART:{group}:PATH_TIME:Update', handler=PVHandler(self.update))

        logging.warning(f'Read PathTimer:latched for group {group} : {self._pathTimer.latched.get()}')

    def update(self, pv, val):
        timev = divmod(float(time.time_ns()), 1.0e9)
        latch = self._pathTimer.latched.get()
        pathtm = [0 for i in range(14)]
        for i in range(14):
            pathtm[i] = self._pathTimer.chan[i].get()
        updatePv(self._pv_latched, latch, timev)
        updatePv(self._pv_array, pathtm, timev)

class PVStats(object):
    def __init__(self, p, m, name, xpm, fiducialPeriod, axiv, hasSfp=True,nAMCs=2,noTiming=False,fidRate=13e6/14.):
        setProvider(p)
        global provider
        provider = p
        global lock
        lock     = m
        global fidPeriod
        fidPeriod  = fiducialPeriod

        self._name = name
        self._xpm  = xpm
        self._app  = xpm.XpmApp

        self.paddr   = addPV(name+':PAddr'  ,'I',self._app.paddr.get())
        self.fwbuild = addPV(name+':FwBuild','s',axiv.BuildStamp.get())
        self.usRxEn  = addPV(name+':UsRxEnable','I',self._app.usRxEnable.get())
        self.cuRxEn  = addPV(name+':CuRxEnable','I',self._app.cuRxEnable.get())

        self._links = []
        for i in range(32):
            self._links.append(LinkStatus(name,self._app,i))

        self._amcPll = []
        for i in range(2):
            self._amcPll.append(AmcPLLStatus(name,self._app,i))

        self._groups = []
        for i in range(8):
            self._groups.append(GroupStats(name+':PART:%d'%i,self._app,i))

        self._pattern = PatternStats(name+':PATT')
        self._usTiming = TimingStatus(name+':Us',xpm.UsTiming,self.usLinkUp,fidRate)
        self._cuTiming = TimingStatus(name+':Cu',xpm.CuTiming,self.cuLinkUp,360.)

        if not noTiming:
            #  Expose for dumping the input link locking status
            self._usTimingLock = TimingLock(name+':Us',xpm.UsGthRx)
            self._cuTimingLock = TimingLock(name+':Cu',xpm.CuGthRx)

        self._cuGen    = CuStatus(name+':XTPG',xpm.CuGenerator,xpm.CuToScPhase)

        self._monClks  = MonClkStatus(name,self._app)
        if hasSfp:
            self._sfpStat  = SFPStatus   (name+':SFPSTATUS',self._xpm,7*nAMCs)
        else:
            self._sfpStat  = QSFPStatus  (name+':QSFPSTATUS',self._xpm)

        self._pathTimer = [PathTimer(self._name, self._xpm, i) for i in range(8)]

#        self._mmcm = []
#        for i,m in enumerate(xpm.mmcms):
#            self._mmcm.append(PVMmcmPhaseLock(name+':XTPG:MMCM%d'%i,m))

    def init(self):
        pass

    def cuLinkUp(self):
        pass

    def usLinkUp(self):
        self.updatePaddr()

    def updatePaddr(self):
        v = self._app.paddr.get()
        logging.info(f'-- updating PADDR to {v}')
        pvUpdate(self.paddr,v)

    def handle(self, msg):
        timev = divmod(float(time.time_ns()), 1.0e9)
        offset = 4
        for i in range(14):
            offset = self._links[i].handle(msg,offset,timev)
        for i in range(8):
            offset = self._groups[i].handle(msg,offset,timev)
        offset = self._pattern.handle(msg,offset,timev)
        for i in range(2):
            offset = self._amcPll[i].handle(msg,offset,timev)
        offset = self._monClks.handle(msg,offset,timev)
        return offset

    def update(self, cycle, noTiming=False, cuMode=False):
        try:
            if noTiming:
                pass
            elif cuMode:
                self._cuTiming.update()
                self._cuGen   .update()
            else:
                self._usTiming.update()
            if self._sfpStat:
                self._sfpStat .update()

#            self.updatePaddr()
        except:
            exc = sys.exc_info()
            if exc[0]==KeyboardInterrupt:
                raise
            else:
                traceback.print_exception(exc[0],exc[1],exc[2])
                logging.error('Caught exception... retrying.')
