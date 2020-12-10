import sys
import time
import traceback
import struct
from p4p.nt import NTScalar
from p4p.nt import NTTable
from p4p.server.thread import SharedPV
from psdaq.pyxpm.pvhandler import *

provider = None
lock     = None

FID_PERIOD    = 1400e-6/1300.
FID_PERIOD_NS = 1400e3 /1300.

def addPV(name,ctype,init=0):
    pv = SharedPV(initial=NTScalar(ctype).wrap(init), handler=DefaultPVHandler())
    provider.add(name, pv)
    return pv
                                
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

def addPVT(name,t):
    table,n = toTable(t)
    init    = toDictList(t,n)
    pv = SharedPV(initial=NTTable(table).wrap(init),
                  handler=DefaultPVHandler())
    provider.add(name,pv)
    return pv

def updatePv(pv,v,timev):
    if v is not None:
        value = pv.current()
        value['value'] = v
        value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
        pv.post(value)

NFPLINKS = 14
sfpStatus  = {'LossOfSignal' : ('ai',[0]*NFPLINKS),
              'ModuleAbsent' : ('ai',[0]*NFPLINKS),
              'TxPower'      : ('af',[0]*NFPLINKS),
              'RxPower'      : ('af',[0]*NFPLINKS)}

class SFPStatus(object):

    def __init__(self, name, xpm):
        self._xpm   = xpm
        self._pv    = addPVT(name,sfpStatus)
        self._value = toDict(sfpStatus)
        self._link  = 0

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
                (self._value['TxPower'][self._link],
                 self._value['RxPower'][self._link]) = amc.SfpI2c.get_pwr()

        self._link += 1
        if self._link==14:
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
        self._rxRcv = self._app.dsLinkRxCnt.get()
        self._rxErr = self._app.dsLinkStatus.get()&0xffff

        def addPVI(label):
            return addPV(name+':'+label+'%d'%i,'I')

        self._pv_txReady      = addPVI('LinkTxReady')
        self._pv_rxReady      = addPVI('LinkRxReady')
        self._pv_txResetDone  = addPVI('LinkTxResetDone')
        self._pv_rxResetDone  = addPVI('LinkRxResetDone')
        self._pv_rxRcv        = addPVI('LinkRxRcv')
        self._pv_rxErr        = addPVI('LinkRxErr')
        self._pv_rxIsXpm      = addPVI('LinkRxIsXpm')
        self._pv_remoteLinkId = addPVI('RemoteLinkId')

    def handle(self,msg,offset,timev):
        w = struct.unpack_from('<LLL',msg,offset)
        offset += 12
        u = (w[2]<<64) + (w[1]<<32) + w[0]
        updatePv(self._pv_txResetDone,(u>>0)&1,timev)
        updatePv(self._pv_txReady    ,(u>>1)&1,timev)
        updatePv(self._pv_rxResetDone,(u>>2)&1,timev)
        updatePv(self._pv_rxReady    ,(u>>3)&1,timev)

        v = (u>>5)&0xffff
        updatePv(self._pv_rxErr,v-self._rxErr,timev)
        self._rxErr = v

        v = (u>>21)&0xffffffff
        updatePv(self._pv_rxRcv,v-self._rxRcv,timev)
        self._rxRcv = v

        updatePv(self._pv_rxIsXpm,(u>>53)&1,timev)
        updatePv(self._pv_remoteLinkId, (u>>54)&0xffffffff,timev)
        return offset

class TimingStatus(object):
    def __init__(self, name, device):
        self._name = name
        self._device = device
        self._device.update()
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

        def addPVF(label):
            return addPV(name+':'+label,'f')

        self._pv_rxClkCount  = addPVF('RxClks')
        self._pv_txClkCount  = addPVF('TxClks')
        self._pv_rxRstCount  = addPVF('RxRsts')
        self._pv_crcErrCount = addPVF('CrcErrs')
        self._pv_rxDecErrs   = addPVF('RxDecErrs')
        self._pv_rxDspErrs   = addPVF('RxDspErrs')
        self._pv_bypassRsts  = addPVF('BypassRsts')
        self._pv_bypassDones = addPVF('BypassDones')
        self._pv_rxLinkUp    = addPVF('RxLinkUp')
        self._pv_fids        = addPVF('FIDs')
        self._pv_sofs        = addPVF('SOFs')
        self._pv_eofs        = addPVF('EOFs')
        self._pv_rxAlign     = addPV(name+':RxAlign', 'aI', [0]*65) 

    def update(self):

        def updatePv(pv,nv,ov,verbose=False,nb=32):
            if nv is not None:
                value = pv.current()
                value['value'] = (nv-ov)&((1<<nb)-1)
                value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
                pv.post(value)
                return nv
            else:
                return ov

        timev = divmod(float(time.time_ns()), 1.0e9)

        self._device.update()
        self._rxClkCount      = updatePv(self._pv_rxClkCount, self._device.RxClkCount.get()<<4, self._rxClkCount)
        self._txClkCount      = updatePv(self._pv_txClkCount, self._device.TxClkCount.get()<<4, self._txClkCount)
        self._rxRstCount      = updatePv(self._pv_rxRstCount, self._device.RxRstCount.get(), self._rxRstCount)
        self._crcErrCount     = updatePv(self._pv_crcErrCount, self._device.CrcErrCount.get(), self._crcErrCount)
        self._rxDecErrCount   = updatePv(self._pv_rxDecErrs, self._device.RxDecErrCount.get(), self._rxDecErrCount)
        self._rxDspErrCount   = updatePv(self._pv_rxDspErrs, self._device.RxDspErrCount.get(), self._rxDspErrCount)
        self._bypassRstCount  = updatePv(self._pv_bypassRsts, self._device.BypassResetCount.get(), self._bypassRstCount)
        self._bypassDoneCount = updatePv(self._pv_bypassDones, self._device.BypassDoneCount.get(), self._bypassDoneCount)
        self._fidCount        = updatePv(self._pv_fids, self._device.FidCount.get(), self._fidCount)
        self._sofCount        = updatePv(self._pv_sofs, self._device.sofCount.get(), self._sofCount)
        self._eofCount        = updatePv(self._pv_eofs, self._device.eofCount.get(), self._eofCount)

        v = self._device.RxLinkUp.get()
        if v is not None:
            value = self._pv_rxLinkUp.current()
            value['value'] = v
            value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
            self._pv_rxLinkUp.post(value)

class AmcPLLStatus(object):
    def __init__(self, name, app, idx):
        self._idx    = idx
        self._idxreg = app.amc
        self._device = app.amcPLL

        def addPVI(label):
            return addPV(name+':'+label+'%d'%idx,'I')

        self._pv_lol    = addPVI('PLL_LOL')
        self._pv_lolCnt = addPVI('PLL_LOLCNT')
        self._pv_los    = addPVI('PLL_LOS')
        self._pv_losCnt = addPVI('PLL_LOSCNT')

    def handle(self, msg, offset, timev):
        w = struct.unpack_from('<B',msg,offset)
        offset += 1
        updatePv(self._pv_lolCnt,(w[0]>>0)&7, timev)
        updatePv(self._pv_lol   ,(w[0]>>3)&1, timev)
        updatePv(self._pv_losCnt,(w[0]>>4)&7, timev)
        updatePv(self._pv_los   ,(w[0]>>7)&1, timev)
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
        updatePv(self._pv_fiducialErr , self._device.cuFiducialIntvErr.get())
        updatePv(self._pv_PhCuToSC    , self._phase .phase())

class MonClkStatus(object):
    def __init__(self, name, app):
        self._app   = app

        self._pv_bpClk  = addPV(name+':BpClk' ,'f')
        self._pv_fbClk  = addPV(name+':FbClk' ,'f')
        self._pv_recClk = addPV(name+':RecClk','f')

        print('MonClkStatus: refClk {:} MHz  recClk {:} MHz'.format(self._app.monClk_1.get()*1.e-6,
                                                                    self._app.monClk_2.get()*1.e-6))

    def handle(self, msg, offset, timev):
        w = struct.unpack_from('<LLL',msg,offset)
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
        l0Stats        = self._app.l0Stats.get()
        self._l0Ena    = self._app.l0EnaCnt(l0Stats)
        self._l0Inh    = self._app.l0InhCnt(l0Stats)
        self._numL0    = self._app.numL0   (l0Stats)
        self._numL0Acc = self._app.numL0Acc(l0Stats)
        self._numL0Inh = self._app.numL0Inh(l0Stats)
        self._linkInhEv = None
        self._linkInhTm = None

        def addPVF(label):
            return addPV(name+':'+label,'f')

        self._pv_runTime   = addPVF('RunTime')
        self._pv_msgDelay  = addPVF('MsgDelay')
        self._pv_l0InpRate = addPVF('L0InpRate')
        self._pv_l0AccRate = addPVF('L0AccRate')
        self._pv_l1Rate    = addPVF('L1Rate')
        self._pv_numL0Inp  = addPVF('NumL0Inp')
        self._pv_numL0Inh  = addPVF('NumL0Inh')
        self._pv_numL0Acc  = addPVF('NumL0Acc')
        self._pv_numL1     = addPVF('NumL1')
        self._pv_deadFrac  = addPVF('DeadFrac')
        self._pv_deadTime  = addPVF('DeadTime')

        self._pv_deadFLink = addPV(name+':DeadFLnk','af',[0.]*32)

    def handle(self,msg,offset,timev):
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
        updatePv(self._pv_runTime , l0Ena*FID_PERIOD, timev)
        updatePv(self._pv_msgDelay, self._app.l0Delay.get(), timev)

        if self._master:
            dL0Ena   = l0Ena    - self._l0Ena
            dL0Inh   = l0Inh    - self._l0Inh
            dt       = dL0Ena*FID_PERIOD
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

            self._l0Ena   = l0Ena
            self._l0Inh   = l0Inh
            self._numL0   = numL0
            self._numL0Acc= numL0Acc
            self._numL0Inh= numL0Inh

        if True:
            if self._linkInhTm:
                den = FID_PERIOD
                linkInhTmV = []
                for i in range(32):
                    linkInhTmV.append((linkInhTm[i] - self._linkInhTm[i])*den)
            else:
                linkInhTmV = [0 for i in range(32)]
            self._linkInhTm = linkInhTm
            updatePv(self._pv_deadFLink, linkInhTmV, timev)

        return offset

class PVMmcmPhaseLock(object):
    def __init__(self, name, mmcm):
        v = []
        v.append( mmcm.delayValue.get() )
        v.append( mmcm.waveform.get() )
        self.pv   = addPV(name,'ai',v)
        

class PVStats(object):
    def __init__(self, p, m, name, xpm):
        global provider
        provider = p
        global lock
        lock     = m

        self._xpm  = xpm
        self._app  = xpm.XpmApp
        
        self._links = []
        for i in range(32):
            self._links.append(LinkStatus(name,self._app,i))

        self._amcPll = []
        for i in range(2):
            self._amcPll.append(AmcPLLStatus(name,self._app,i))

        self._groups = []
        for i in range(8):
            self._groups.append(GroupStats(name+':PART:%d'%i,self._app,i))

        self._usTiming = TimingStatus(name+':Us',xpm.UsTiming)
        self._cuTiming = TimingStatus(name+':Cu',xpm.CuTiming)
        self._cuGen    = CuStatus(name+':XTPG',xpm.CuGenerator,xpm.CuToScPhase)
        self._monClks  = MonClkStatus(name,self._app)
        self._sfpStat  = SFPStatus   (name+':SFPSTATUS',self._xpm)

        self.paddr   = addPV(name+':PAddr'  ,'I',self._app.paddr.get())
        self.fwbuild = addPV(name+':FwBuild','s',self._xpm.AxiVersion.BuildStamp.get())

#        self._mmcm = []
#        for i,m in enumerate(xpm.mmcms):
#            self._mmcm.append(PVMmcmPhaseLock(name+':XTPG:MMCM%d'%i,m))

    def init(self):
        pass

    def handle(self, msg):
        timev = divmod(float(time.time_ns()), 1.0e9)
        offset = 4
        for i in range(14):
            offset = self._links[i].handle(msg,offset,timev)
        for i in range(8):
            offset = self._groups[i].handle(msg,offset,timev)
        for i in range(2):
            offset = self._amcPll[i].handle(msg,offset,timev)
        offset = self._monClks.handle(msg,offset,timev)

    def update(self):
        try:
            self._usTiming.update()
            self._cuTiming.update()
            self._cuGen   .update()
        except:
            exc = sys.exc_info()
            if exc[0]==KeyboardInterrupt:
                raise
            else:
                traceback.print_exception(exc[0],exc[1],exc[2])
                print('Caught exception... retrying.')
