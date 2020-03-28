import sys
import time
import traceback
import threading
import socket
import struct
from p4p.nt import NTScalar
from p4p.server.thread import SharedPV
from psdaq.pyxpm.pvseq import *
from psdaq.pyxpm.pvhandler import *

provider = None
lock     = None

class TransitionId(object):
    Clear   = 0
    Config  = 2
    Enable  = 4
    Disable = 5

class RateSel(object):
    FixedRate = 0
    ACRate    = 1
    EventCode = 2
    Sequence  = 3

def pipelinedepth_from_delay(value):
    v = value &0xffff
    return ((v*200)&0xffff) | (v<<16)

def forceUpdate(reg):
    reg.get()

def retry(cmd,pv,value):
    for tries in range(5):
        try:
            cmd(pv,value)
            break
        except:
            exc = sys.exc_info()
            if exc[0]==KeyboardInterrupt:
                raise
            else:
                traceback.print_exception(exc[0],exc[1],exc[2])
                print('Caught exception... retrying.')

def retry_wlock(cmd,pv,value):
    lock.acquire()
    retry(cmd,pv,value)
    lock.release()

class RetryWLock(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, pv, value):
        retry_wlock(self.func, pv, value)

class RegH(PVHandler):
    def __init__(self, valreg):
        super(RegH,self).__init__(self.handle)
        self._valreg   = valreg

    def cmd(self,pv,value):
        self._valreg.post(value)

    def handle(self, pv, value):
        retry(self.cmd,pv,value)

    
class IdxRegH(PVHandler):
    def __init__(self, valreg, idxreg, idx):
        super(IdxRegH,self).__init__(self.handle)
        self._valreg   = valreg
        self._idxreg   = idxreg
        self._idx      = idx

    def cmd(self,pv,value):
        self._idxreg.post(self._idx)
        forceUpdate(self._valreg)
        self._valreg.post(value)

    def handle(self, pv, value):
        retry_wlock(self.cmd,pv,value)

class L0DelayH(IdxRegH):
    def __init__(self, valreg, idxreg, idx):
        super(L0DelayH,self).__init__(valreg, idxreg, idx)

    def handle(self, pv, value):
        retry_wlock(self.cmd,pv,pipelinedepth_from_delay(value))

class CmdH(PVHandler):

    def __init__(self, cmd):
        super(CmdH,self).__init__(self.handle)
        self._cmd      = cmd

    def cmd(self,pv,value):
        self._cmd()

    def handle(self, pv, value):
        if value:
            retry(self.cmd,pv,value) 
            
class IdxCmdH(PVHandler):

    def __init__(self, cmd, idxcmd, idx):
        super(IdxCmdH,self).__init__(self.handle)
        self._cmd      = cmd
        self._idxcmd   = idxcmd
        self._idx      = idx

    def cmd(self,pv,value):
        self._idxcmd.set(self._idx)
        self._cmd()

    def handle(self, pv, value):
        if value:
            retry_wlock(self.idxcmd,pv,value)
            
class RegArrayH(PVHandler):

    def __init__(self, valreg):
        super(RegArrayH,self).__init__(self.handle)
        self._valreg   = valreg

    def cmd(self,pv,val):
        for reg in self._valreg.values():
            reg.post(val)
        
    def handle(self, pv, val):
        retry(self.cmd,pv,val)

class LinkCtrls(object):
    def __init__(self, name, xpm, link):
        self._link  = link
        self._ringb = xpm.AxiLiteRingBuffer
        self._app   = xpm.XpmApp

        app = self._app
        linkreg = app.link

        def addPV(label, reg, init=0):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=IdxRegH(reg,linkreg,link))
            provider.add(name+':'+label+'%d'%link,pv)
            reg.set(init)  #  initialization
            return pv

        linkreg.set(link)  #  initialization
        app.fullEn.set(1)
#        self._pv_linkRxTimeout = addPV('LinkRxTimeout',app.rxTimeout, init=186)
        self._pv_linkGroupMask = addPV('LinkGroupMask',app.fullMask)
#        self._pv_linkTrigSrc   = addPV('LinkTrigSrc'  ,app.trigSrc)
        self._pv_linkLoopback  = addPV('LinkLoopback' ,app.loopback)
        self._pv_linkTxReset   = addPV('TxLinkReset'  ,app.txReset)
        self._pv_linkRxReset   = addPV('RxLinkReset'  ,app.rxReset)
        print('LinkCtrls.init link {}  fullEn {}'
              .format(link, app.fullEn.get()))
        
        def addPV(label, init, handler):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=handler)
            provider.add(name+':'+label+'%d'%link,pv)
            return pv

        self._pv_linkRxDump    = addPV('RxLinkDump',0,handler=PVHandler(self.dump))

    def _dump(self, pv, val):
        self._app.linkDebug.set(self._link)
        self._ringb.clear.set(1)
        time.sleep(1e-6)
        self._ringb.clear.set(0)
        self._ringb.start.set(1)
        time.sleep(100e-6)
        self._ringb.start.set(0)
            
    def dump(self,pv,val):
        if val:
            retry_wlock(self._dump,pv,val)
            self._ringb.Dump()

class CuGenCtrls(object):
    def __init__(self, name, xpm):

        def addPV(label, init, reg):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=RegH(reg))
            provider.add(name+':'+label,pv)
            reg.set(init)
            return pv

        self._pv_cuDelay    = addPV('CuDelay'   , 200*800, xpm.CuGenerator.cuDelay)
        self._pv_cuBeamCode = addPV('CuBeamCode',     140, xpm.CuGenerator.cuBeamCode)
        self._pv_clearErr   = addPV('ClearErr'  ,       0, xpm.CuGenerator.cuFiducialIntvErr)

        def addPV(label, init, reg):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=RegArrayH(reg))
            provider.add(name+':'+label,pv)
            for r in reg.values():
                r.set(init)
            return pv

        self._pv_cuInput    = addPV('CuInput'   ,       2, xpm.AxiSy56040.OutputConfig)

class PVInhibit(object):
    def __init__(self, name, app, inh, group, idx):
        self._group = group
        self._idx   = idx
        self._app   = app
        self._inh   = inh

        def addPV(label,cmd,init):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=PVHandler(cmd))
            provider.add(name+':'+label+'%d'%idx,pv)
            cmd(pv,init)  # initialize
            return pv

        self._pv_InhibitInt = addPV('InhInterval', RetryWLock(self.inhibitIntv), 10)
        self._pv_InhibitLim = addPV('InhLimit'   , RetryWLock(self.inhibitLim ),  4)
        self._pv_InhibitEna = addPV('InhEnable'  , RetryWLock(self.inhibitEna ),  0)

    def inhibitIntv(self, pv, value):
        self._app.partition.set(self._group)
        forceUpdate(self._inh.intv)
        self._inh.intv.set(value-1)

    def inhibitLim(self, pv, value):
        self._app.partition.set(self._group)
        forceUpdate(self._inh.maxAcc)
        self._inh.maxAcc.set(value-1)

    def inhibitEna(self, pv, value):
        self._app.partition.set(self._group)
        forceUpdate(self._inh.inhEn)
        self._inh.inhEn.set(value)

class GroupSetup(object):
    def __init__(self, name, app, group, stats):
        self._group = group
        self._app   = app
        self._stats = stats

        def addPV(label,cmd,init=0,set=False):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=PVHandler(cmd))
            provider.add(name+':'+label,pv)
            if set:
                cmd(pv,init)
            return pv

        self._pv_L0Select   = addPV('L0Select'               ,self.put)
        self._pv_FixedRate  = addPV('L0Select_FixedRate'     ,self.put)
        self._pv_ACRate     = addPV('L0Select_ACRate'        ,self.put)
        self._pv_ACTimeslot = addPV('L0Select_ACTimeslot'    ,self.put)
        self._pv_EventCode  = addPV('L0Select_EventCode'     ,self.put)
        self._pv_Sequence   = addPV('L0Select_Sequence'      ,self.put)
        self._pv_SeqBit     = addPV('L0Select_SeqBit'        ,self.put)        
        self._pv_DstMode    = addPV('DstSelect'              ,self.put, 1)
        self._pv_DstMask    = addPV('DstSelect_Mask'         ,self.put)
        self._pv_Run        = addPV('Run'                    ,self.run    , set=True)
        self._pv_Master     = addPV('Master'                 ,self.master, set=True)

        def addPV(label,reg,init=0,set=False):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=IdxRegH(reg,self._app.partition,group))
            provider.add(name+':'+label,pv)
            if set:
                self._app.partition.set(group)
                reg.set(init)
            return pv
        
        self._pv_MsgHeader  = addPV('MsgHeader' , app.msgHdr ,  0, set=True)
        self._pv_MsgPayload = addPV('MsgPayload', app.msgPayl,  0, set=True)

        def addPV(label,reg,init=0,set=False):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=L0DelayH(reg,self._app.partition,group))
            provider.add(name+':'+label,pv)
            if set:
                self._app.partition.set(group)
                reg.set(pipelinedepth_from_delay(init))
            return pv

        self._pv_L0Delay    = addPV('L0Delay'   , app.pipelineDepth, 90, set=True)

        #  initialize
        self.put(None,None)

        def addPV(label):
            pv = SharedPV(initial=NTScalar('I').wrap(0), 
                          handler=DefaultPVHandler())
            provider.add(name+':'+label,pv)
            return pv

        self._pv_MsgConfigKey = addPV('MsgConfigKey')

        self._inhibits = []
##  Remove temporarily while we test Ben's xpm
        self._inhibits.append(PVInhibit(name, app, app.inh_0, group, 0))
        self._inhibits.append(PVInhibit(name, app, app.inh_1, group, 1))
        self._inhibits.append(PVInhibit(name, app, app.inh_2, group, 2))
        self._inhibits.append(PVInhibit(name, app, app.inh_3, group, 3))

    def dump(self):
        print('Group: {}  Master: {}  RateSel: {:x}  DestSel: {:x}  Ena: {}'
              .format(self._group, self._app.l0Master.get(), self._app.l0RateSel.get(), self._app.l0DestSel.get(), self._app.l0En.get()))

    def setFixedRate(self):
        rateVal = (0<<14) | (self._pv_FixedRate.current()['value']&0xf)
        self._app.l0RateSel.set(rateVal)
        
    def setACRate(self):
        acRate = self._pv_ACRate    .current()['value']
        acTS   = self._pv_ACTimeslot.current()['value']
        rateVal = (1<<14) | ((acTS&0x3f)<<3) | (acRate&0x7)
        self._app.l0RateSel.set(rateVal)

    def setEventCode(self):
        code   = self._pv_EventCode.current()['value']
        rateVal = (2<<14) | ((code&0xf0)<<4) | (code&0xf)
        self._app.l0RateSel.set(rateVal)

    def setSequence(self):
        seqIdx = self._pv_Sequence.current()['value']
        seqBit = self._pv_SeqBit  .current()['value']
        rateVal = (2<<14) | ((seqIdx&0x3f)<<8) | (seqBit&0xf)
        self._app.l0RateSel.set(rateVal)

    def setDestn(self):
        mode = self._pv_DstMode.current()['value']
        mask = self._pv_DstMask.current()['value']
        destVal  = (mode<<15) | (mask&0x7fff)
        self._app.l0DestSel.set(destVal)

    def master(self, pv, val):
        lock.acquire()
        self._app.partition.set(self._group)
        forceUpdate(self._app.l0Master)

        if val==0:
            self._app.l0Master.set(0)
            self._app.l0En    .set(0)
            self._stats._master = 0
            
            curr = self._pv_Run.current()
            curr['value'] = 0
            self._pv_Run.post(curr)
        else:
            self._app.l0Master.set(1)
            self._stats._master = 1
        lock.release()

    def put(self, pv, val):
        lock.acquire()
        self._app.partition.set(self._group)
        forceUpdate(self._app.l0RateSel)
        mode = self._pv_L0Select.current()['value']
        if mode == RateSel.FixedRate:
            self.setFixedRate()
        elif mode == RateSel.ACRate:
            self.setACRate()
        elif mode == RateSel.EventCode:
            self.setEventCode()
        elif mode == RateSel.Sequence:
            self.setSequence()
        else:
            print('L0Select mode invalid {}'.format(mode))

        forceUpdate(self._app.l0DestSel)
        self.setDestn()
        self.dump()
        lock.release()
            
    def run(self, pv, val):
        lock.acquire()
        self._app.partition.set(self._group)
        forceUpdate(self._app.l0En)
        enable = 1 if val else 0
        self._app.l0En.set(enable)
        self.dump()
        lock.release()


class GroupCtrls(object):
    def __init__(self, name, app, stats):

        def addPV(label,reg):
            pv = SharedPV(initial=NTScalar('I').wrap(0), 
                          handler=RegH(reg))
            provider.add(name+':'+label,pv)
            return pv

        self._pv_l0Reset   = addPV('GroupL0Reset'  ,app.groupL0Reset)
        self._pv_l0Enable  = addPV('GroupL0Enable' ,app.groupL0Enable)
        self._pv_l0Disable = addPV('GroupL0Disable',app.groupL0Disable)
        self._pv_MsgInsert = addPV('GroupMsgInsert',app.groupMsgInsert)

        self._groups = []
        for i in range(8):
            self._groups.append(GroupSetup(name+':PART:%d'%i, app, i, stats[i]))

        #  This is necessary in XTPG
        app.groupL0Reset.set(0xff)
        app.groupL0Reset.set(0)

class PVCtrls(object):

    def __init__(self, p, m, name, ip, xpm, stats):
        global provider
        provider = p
        global lock
        lock     = m

        # Assign transmit link ID
        ip_comp = ip.split('.')
        xpm_num = name.rsplit(':',1)[1]
        v = 0xff00000 | ((int(xpm_num)&0xf)<<16) | ((int(ip_comp[2])&0xf)<<12) | ((int(ip_comp[3])&0xff)<< 4)
        xpm.XpmApp.paddr.set(v)
        print('Set PADDR to 0x{:x}'.format(v))

        self._ip    = ip

        self._links = []
        for i in range(24):
            self._links.append(LinkCtrls(name, xpm, i))

        app = xpm.XpmApp

        self._pv_amcDumpPLL = []
        for i in range(2):
            pv = SharedPV(initial=NTScalar('I').wrap(0), 
                          handler=IdxCmdH(app.amcPLL.Dump,app.amc,i))
            provider.add(name+':DumpPll%d'%i,pv)
            self._pv_amcDumpPLL.append(pv)

        self._cu    = CuGenCtrls(name+':XTPG', xpm)

        self._group = GroupCtrls(name, app, stats)

##  Remove sequencer while we test Ben's image
        if True:
            self._seq = PVSeq(provider, name+':SEQENG:0', ip, Engine(0, xpm.SeqEng_0))

            self._pv_dumpSeq = SharedPV(initial=NTScalar('I').wrap(0), 
                                        handler=CmdH(self._seq._eng.dump))
            provider.add(name+':DumpSeq',self._pv_dumpSeq)

        self._pv_usRxReset = SharedPV(initial=NTScalar('I').wrap(0),
                                      handler=CmdH(xpm.UsTiming.C_RxReset))
        provider.add(name+':Us:RxReset',self._pv_usRxReset)

        self._pv_cuRxReset = SharedPV(initial=NTScalar('I').wrap(0),
                                      handler=CmdH(xpm.CuTiming.C_RxReset))
        provider.add(name+':Cu:RxReset',self._pv_cuRxReset)

        self._thread = threading.Thread(target=self.notify)
        self._thread.start()


    def notify(self):
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.connect((self._ip,8197))
        
        msg = b'\x00\x00\x00\x00'
        client.send(msg)
        
        while True:
            msg = client.recv(256)
            s = struct.Struct('H')
            siter = s.iter_unpack(msg)
            mask = next(siter)[0]
            print('mask {:x}'.format(mask))
            i=0
            while mask!=0:
                if mask&1:
                    addr = next(siter)[0]
                    print('addr[{}] {:x}'.format(i,addr))
                    if i<1:
                        self._seq.checkPoint(addr)
                i += 1
                mask = mask>>1
            
            
