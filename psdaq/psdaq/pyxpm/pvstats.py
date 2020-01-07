import sys
import time
import traceback
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

    def update(self):

        def updatePv(pv,v):
            if v is not None:
                value = pv.current()
                value['value'] = v
                value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
                pv.post(value)

        timev = divmod(float(time.time_ns()), 1.0e9)
        lock.acquire()
        self._app.link.set(self._idx)
        status = self._app.dsLinkStatus.get()
        if status is not None:
            updatePv(self._pv_txResetDone,(status>>16)&1)
            updatePv(self._pv_txReady    ,(status>>17)&1)
            updatePv(self._pv_rxResetDone,(status>>18)&1)
            updatePv(self._pv_rxReady    ,(status>>19)&1)
            updatePv(self._pv_rxIsXpm    ,(status>>20)&1)
            if self._idx==16:
                pass
            else:
                value = status&0xffff
                updatePv(self._pv_rxErr,value-self._rxErr)
                self._rxErr = value
        value = self._app.dsLinkRxCnt.get()
        if value is not None:
            updatePv(self._pv_rxRcv,value-self._rxRcv)
            self._rxRcv = value
        updatePv(self._pv_remoteLinkId,self._app.remId.get())
        lock.release()

class TimingStatus(object):
    def __init__(self, name, device):
        self._device = device
        self._rxClkCount      = device.RxClkCount.get()
        self._txClkCount      = device.TxClkCount.get()
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
        self._pv_rxRstCount  = addPVF('RxDecErrs')
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

        self._rxClkCount      = updatePv(self._pv_rxClkCount, self._device.RxClkCount.get(), self._rxClkCount)
        self._txClkCount      = updatePv(self._pv_txClkCount, self._device.TxClkCount.get(), self._txClkCount)
        self._rxRstCount      = updatePv(self._pv_rxRstCount, self._device.RxRstCount.get(), self._rxRstCount)
        self._crcErrCount     = updatePv(self._pv_crcErrCount, self._device.CrcErrCount.get(), self._crcErrCount)
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

    def update(self):

        def updatePv(pv,v):
            if v is not None:
                value = pv.current()
                value['value'] = v
                value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
                pv.post(value)

        timev = divmod(float(time.time_ns()), 1.0e9)
        lock.acquire()
        self._idxreg.set(self._idx)
        updatePv(self._pv_lol   ,self._device.lol   .get())
        updatePv(self._pv_lolCnt,self._device.lolCnt.get())
        updatePv(self._pv_los   ,self._device.los   .get())
        updatePv(self._pv_losCnt,self._device.losCnt.get())
        lock.release()

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

    def update(self):

        def updatePv(pv,v):
            if v is not None:
                value = pv.current()
                value['value'] = v
                value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
                pv.post(value)

        timev = divmod(float(time.time_ns()), 1.0e9)
        updatePv(self._pv_bpClk , self._app.monClk_0.get())
        updatePv(self._pv_fbClk , self._app.monClk_1.get())
        updatePv(self._pv_recClk, self._app.monClk_2.get())

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
        linkInhEvReg   = self._app.inhEvCnt.get()
        linkInhTmReg   = self._app.inhTmCnt.get()
        self._linkInhEv = []
        self._linkInhTm = []
        for i in range(32):
            self._linkInhEv.append((linkInhEvReg>>(32*i))&0xffffffff)
            self._linkInhTm.append((linkInhTmReg>>(32*i))&0xffffffff)

        def addPVF(label):
            return addPV(name+':'+label,'f')

        self._pv_runTime   = addPVF('RunTime')
        self._pv_msgDelay  = addPVF('MsgDelay')
        self._pv_l0InpRate = addPVF('L0InpRate')
        self._pv_l0AccRate = addPVF('L0AccRate')
        self._pv_l1Rate    = addPVF('L1Rate')
        self._pv_numL0Inp  = addPVF('NumL0Inp')
        self._pv_numL0Acc  = addPVF('NumL0Acc')
        self._pv_numL1     = addPVF('NumL1')
        self._pv_deadFrac  = addPVF('DeadFrac')
        self._pv_deadTime  = addPVF('DeadTime')

        self._pv_deadFLink = addPV(name+':DeadFLnk','af',[0.]*32)

    def update(self):

        def updatePv(pv,v):
            if v is not None:
                value = pv.current()
                value['value'] = v
                value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
                pv.post(value)

        timeval = float(time.time_ns())
        timev = divmod(timeval, 1.0e9)
        lock.acquire()
        self._app.partition.set(self._group)
        if self._master:
            l0Stats  = self._app.l0Stats.get()
            if l0Stats is not None:
                l0Ena    = self._app.l0EnaCnt(l0Stats)
                l0Inh    = self._app.l0InhCnt(l0Stats)
                numL0    = self._app.numL0   (l0Stats)
                numL0Acc = self._app.numL0Acc(l0Stats)
                numL0Inh = self._app.numL0Inh(l0Stats)

                updatePv(self._pv_runTime, l0Ena*FID_PERIOD)
                updatePv(self._pv_msgDelay, self._app.l0Delay.get())
                dL0Ena   = l0Ena    - self._l0Ena
                dL0Inh   = l0Inh    - self._l0Inh
                dt       = dL0Ena*FID_PERIOD
                dnumL0   = numL0    - self._numL0
                dnumL0Acc= numL0Acc - self._numL0Acc
                dnumL0Inh= numL0Inh - self._numL0Inh
                if dL0Ena:
                    l0InpRate = dnumL0/dt
                    l0AccRate = dnumL0Acc/dt
                    updatePv(self._pv_deadTime, dL0Inh/dL0Ena)
                    linkInhEvReg = self._app.inhEvCnt.get()
                    linkInhEv = []
                    for i in range(32):
                        linkInh = (linkInhEvReg>>(32*i))&0xffffffff
                        linkInhEv.append((linkInh - self._linkInhEv[i])/dL0Ena)
                        self._linkInhEv[i] = linkInh
                    updatePv(self._pv_deadFLink, linkInhEv)
                else:
                    l0InpRate = 0
                    l0AccRate = 0
                updatePv(self._pv_l0InpRate, l0InpRate)
                updatePv(self._pv_l0AccRate, l0AccRate)
                updatePv(self._pv_numL0Inp, numL0)
                updatePv(self._pv_numL0Acc, numL0Acc)
                if dnumL0:
                    deadFrac = dnumL0Inh/dnumL0
                else:
                    deadFrac = 0
                updatePv(self._pv_deadFrac, deadFrac)

                self._l0Ena   = l0Ena
                self._l0Inh   = l0Inh
                self._numL0   = numL0
                self._numL0Acc= numL0Acc
                self._numL0Inh= numL0Inh
                
        else:
            nfid = (timeval - self._timeval)/FID_PERIOD_NS
            linkInhTmReg = self._app.inhTmCnt.get()
            if linkInhTmReg is not None:
                linkInhTm = []
                for i in range(32):
                    linkInh = (linkInhTmReg>>(32*i))&0xffffffff
                    linkInhTm.append((linkInh - self._linkInhTm[i])/nfid)
                    self._linkInhTm[i] = linkInh
                updatePv(self._pv_deadFLink, linkInhTm)
        lock.release()

        self._timeval = timeval

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

    def update(self):
        try:
            for i in range(32):
                self._links[i].update()
            for i in range(2):
                self._amcPll[i].update()
            for i in range(8):
                self._groups[i].update()

            self._usTiming.update()
            self._cuTiming.update()
            self._cuGen   .update()
            self._monClks .update()
##  Remove while we test Ben's image
            self._sfpStat .update()
        except:
            exc = sys.exc_info()
            if exc[0]==KeyboardInterrupt:
                raise
            else:
                traceback.print_exception(exc[0],exc[1],exc[2])
                print('Caught exception... retrying.')
