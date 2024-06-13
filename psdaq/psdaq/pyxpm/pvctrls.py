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
# db support
from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
from psdaq.configdb.get_config import get_config_with_params

provider = None
lock     = None
countdn  = 0
countrst = 60
_fidPrescale = 200
_fidPeriod   = 1400/1.3

NCODES = 32
seqCodes = {'EventCode'   : ('ai',[i for i in range(288-NCODES,288)]),
            'Description' : ('as',['']*NCODES),
            'Rate'        : ('ai',[0]*NCODES),
            'Enabled'     : ('ab',[False]*NCODES)}

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
    return ((v*_fidPrescale)&0xffff) | (v<<16)

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
                logging.error('Caught exception... retrying.')

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
    def __init__(self, valreg, archive=False):
        super(RegH,self).__init__(self.handle)
        self._valreg   = valreg
        self._archive  = archive

    def cmd(self,pv,value):
        self._valreg.post(value)

    def handle(self, pv, value):
        global countdn
        retry(self.cmd,pv,value)
        if self._archive:
            countdn = countrst

class CuDelayH(RegH):
    def __init__(self, valreg, archive, pvu):
        super(CuDelayH,self).__init__(valreg,archive)
        self.pvu = pvu

    def handle(self, pv, value):
        super(CuDelayH,self).handle(pv,value)
        curr = self.pvu.current()
        curr['value'] = value*7000./1300
        self.pvu.post(curr)

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
    def __init__(self, valreg, idxreg, idx, pvu):
        super(L0DelayH,self).__init__(valreg, idxreg, idx)
        self.pvu = pvu

    def handle(self, pv, value):
        global countdn
        retry_wlock(self.cmd,pv,pipelinedepth_from_delay(value))

        curr = self.pvu.current()
        curr['value'] = value*_fidPeriod
        self.pvu.post(curr)
        countdn = countrst

class CmdH(PVHandler):

    def __init__(self, cmd):
        super(CmdH,self).__init__(self.handle)
        self._cmd      = cmd

    def cmd(self,pv,value):
        self._cmd()

    def handle(self, pv, value):
        if value:
            retry(self.cmd,pv,value) 
            
class RegArrayH(PVHandler):

    def __init__(self, valreg, archive=False):
        super(RegArrayH,self).__init__(self.handle)
        self._valreg   = valreg
        self._archive  = archive

    def cmd(self,pv,val):
        global countdn
        for reg in self._valreg.values():
            reg.post(val)
        if self._archive:
            countdn = countrst
        
    def handle(self, pv, val):
        retry(self.cmd,pv,val)

class LinkCtrls(object):
    def __init__(self, name, xpm, link):
        self._link  = link
        self._ringb = xpm.AxiLiteRingBuffer
        self._app   = xpm.XpmApp

        app = self._app
        linkreg = app.link

        def _addPV(label, reg, init=0):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=IdxRegH(reg,linkreg,link))
            provider.add(name+':'+label+'%d'%link,pv)
            reg.set(init)  #  initialization
            return pv

        linkreg.set(link)  #  initialization
        app.fullEn.set(1)
#        self._pv_linkRxTimeout = _addPV('LinkRxTimeout',app.rxTimeout, init=186)
        self._pv_linkGroupMask = _addPV('LinkGroupMask',app.fullMask)
#        self._pv_linkTrigSrc   = _addPV('LinkTrigSrc'  ,app.trigSrc)
        self._pv_linkLoopback  = _addPV('LinkLoopback' ,app.loopback)
        self._pv_linkTxReset   = _addPV('TxLinkReset'  ,app.txReset)
        self._pv_linkRxReset   = _addPV('RxLinkReset'  ,app.rxReset)
        logging.info(f'LinkCtrls.init link {link}  fullEn {app.fullEn.get()}')
        
        def _addPV(label, init, handler):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=handler)
            provider.add(name+':'+label+'%d'%link,pv)
            return pv

        self._pv_linkRxDump    = _addPV('RxLinkDump',0,handler=PVHandler(self.dump))

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
    def __init__(self, name, xpm, dbinit=None):

        try:
            cuDelay    = dbinit['XTPG']['CuDelay']
            cuBeamCode = dbinit['XTPG']['CuBeamCode']
            cuInput    = dbinit['XTPG']['CuInput']
            logging.info(f'Read XTPG parameters CuDelay {cuDelay}, CuBeamCode {cuBeamCode}, CuInput {cuInput}')
        except:
            cuDelay    = _fidPrescale*800
            cuBeamCode = 140
            cuInput    = 1
            logging.info('Defaulting XTPG parameters')

        def _addPV(label, init, reg, archive):
            pvu = SharedPV(initial=NTScalar('f').wrap(init*7000./1300), 
                          handler=DefaultPVHandler())
            provider.add(name+':'+label+'_ns',pvu)

            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=CuDelayH(reg,archive,pvu))
            provider.add(name+':'+label,pv)
            reg.set(init)
            return pv

        self._pv_cuDelay    = _addPV('CuDelay'   ,    cuDelay, xpm.CuGenerator.cuDelay          , True)

        def _addPV(label, init, reg, archive):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=RegH(reg,archive=archive))
            provider.add(name+':'+label,pv)
            reg.set(init)
            return pv

        self._pv_cuBeamCode = _addPV('CuBeamCode', cuBeamCode, xpm.CuGenerator.cuBeamCode       , True)
        self._pv_clearErr   = _addPV('ClearErr'  ,          0, xpm.CuGenerator.cuFiducialIntvErr, False)

        def _addPV(label, init, reg, archive):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=RegArrayH(reg, archive=archive))
            provider.add(name+':'+label,pv)
            for r in reg.values():
                r.set(init)
            return pv

        self._pv_cuInput    = _addPV('CuInput'   , cuInput, xpm.AxiSy56040.OutputConfig, True)

class PVInhibit(object):
    def __init__(self, name, app, inh, group, idx):
        self._group = group
        self._idx   = idx
        self._app   = app
        self._inh   = inh

        def _addPV(label,cmd,init):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=PVHandler(cmd))
            provider.add(name+':'+label+'%d'%idx,pv)
            cmd(pv,init)  # initialize
            return pv

        self._pv_InhibitInt = _addPV('InhInterval', RetryWLock(self.inhibitIntv), 10)
        self._pv_InhibitLim = _addPV('InhLimit'   , RetryWLock(self.inhibitLim ),  4)
        self._pv_InhibitEna = _addPV('InhEnable'  , RetryWLock(self.inhibitEna ),  0)

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
    def __init__(self, name, app, group, stats, init=None):
        self._group = group
        self._app   = app
        self._stats = stats

        def _addPV(label,cmd,init=0,set=False):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=PVHandler(cmd))
            provider.add(name+':'+label,pv)
            if set:
                cmd(pv,init)
            return pv

        self._pv_L0Select   = _addPV('L0Select'               ,self.put)
        self._pv_FixedRate  = _addPV('L0Select_FixedRate'     ,self.put)
        self._pv_ACRate     = _addPV('L0Select_ACRate'        ,self.put)
        self._pv_ACTimeslot = _addPV('L0Select_ACTimeslot'    ,self.put)
        self._pv_EventCode  = _addPV('L0Select_EventCode'     ,self.put)
        self._pv_Sequence   = _addPV('L0Select_Sequence'      ,self.put)
        self._pv_SeqBit     = _addPV('L0Select_SeqBit'        ,self.put)        
        self._pv_DstMode    = _addPV('DstSelect'              ,self.put, 1)
        self._pv_DstMask    = _addPV('DstSelect_Mask'         ,self.put)
        self._pv_Run        = _addPV('Run'                    ,self.run   , set=True)
        self._pv_Master     = _addPV('Master'                 ,self.master, set=True)

        self._pv_StepDone   = SharedPV(initial=NTScalar('I').wrap(0), handler=DefaultPVHandler())
        provider.add(name+':StepDone', self._pv_StepDone)

        self._pv_StepGroups = _addPV('StepGroups'            ,self.stepGroups, set=True)
        self._pv_StepEnd    = _addPV('StepEnd'               ,self.stepEnd   , set=True)

        def _addPV(label,reg,init=0,set=False):
            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=IdxRegH(reg,self._app.partition,group))
            provider.add(name+':'+label,pv)
            if set:
                self._app.partition.set(group)
                reg.set(init)
            return pv
        
        self._pv_MsgHeader  = _addPV('MsgHeader' , app.msgHdr ,  0, set=True)
        self._pv_MsgPayload = _addPV('MsgPayload', app.msgPayl,  0, set=True)
        self._pv_L0Groups   = _addPV('L0Groups'  , app.l0Groups, 0, set=True)
        self._pv_RawUpdate  = _addPV('L0RawUpdate', app.l0RawUpdate, 1, set=True)

        def _addPV(label,reg,init=0,set=False):
            pvu = SharedPV(initial=NTScalar('f').wrap(init*_fidPeriod),
                           handler=DefaultPVHandler())
            provider.add(name+':'+label+'_ns',pvu)

            pv = SharedPV(initial=NTScalar('I').wrap(init), 
                          handler=L0DelayH(reg,self._app.partition,group,pvu))
            provider.add(name+':'+label,pv)
            if set:
                self._app.partition.set(group)
                reg.set(pipelinedepth_from_delay(init))
            return pv

        self._pv_L0Delay    = _addPV('L0Delay'   , app.pipelineDepth, init['L0Delay'][group] if init else 90, set=True)

        #  initialize
        self.put(None,None)

        def _addPV(label):
            pv = SharedPV(initial=NTScalar('I').wrap(0), 
                          handler=DefaultPVHandler())
            provider.add(name+':'+label,pv)
            return pv

        self._pv_MsgConfigKey = _addPV('MsgConfigKey')

        self._inhibits = []
        self._inhibits.append(PVInhibit(name, app, app.inh_0, group, 0))
        self._inhibits.append(PVInhibit(name, app, app.inh_1, group, 1))
        self._inhibits.append(PVInhibit(name, app, app.inh_2, group, 2))
        self._inhibits.append(PVInhibit(name, app, app.inh_3, group, 3))

    def dump(self):
        logging.info(f'Group: {self._group}  Master: {self._app.l0Master.get()}  RateSel: {self._app.l0RateSel.get():x}  DestSel: {self._app.l0DestSel.get():x}  Ena: {self._app.l0En.get()}')

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
        rateVal = (2<<14) | ((code&0x3f0)<<4) | (code&0xf)
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
            logging.warning('L0Select mode invalid {}'.format(mode))

        forceUpdate(self._app.l0DestSel)
        self.setDestn()
        #self.dump()
        lock.release()

    def stepGroups(self, pv, val):
        getattr(self._app,'stepGroup%i'%self._group).set(val)

    def stepEnd(self, pv, val):
        self.stepDone(False)
        getattr(self._app,'stepEnd%i'%self._group).set(val)

    def stepDone(self, val):
        value = self._pv_StepDone.current()
        value['value'] = 1 if val else 0
        timev = divmod(float(time.time_ns()), 1.0e9)
        value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
        self._pv_StepDone.post(value)

    def run(self, pv, val):
        lock.acquire()
        self._app.partition.set(self._group)
        forceUpdate(self._app.l0En)
        enable = 1 if val else 0
        self._app.l0En.set(enable)
        self.dump()
        lock.release()


class GroupCtrls(object):
    def __init__(self, name, app, stats, init=None):

        def _addPV(label,reg):
            pv = SharedPV(initial=NTScalar('I').wrap(0), 
                          handler=RegH(reg))
            provider.add(name+':'+label,pv)
            return pv

        self._pv_l0Reset   = _addPV('GroupL0Reset'  ,app.groupL0Reset)
        self._pv_l0Enable  = _addPV('GroupL0Enable' ,app.groupL0Enable)
        self._pv_l0Disable = _addPV('GroupL0Disable',app.groupL0Disable)
        self._pv_MsgInsert = _addPV('GroupMsgInsert',app.groupMsgInsert)

        self._groups = []
        for i in range(8):
            self._groups.append(GroupSetup(name+':PART:%d'%i, app, i, stats[i], init=init['PART'] if init else None))

        #  This is necessary in XTPG
        app.groupL0Reset.set(0xff)
        app.groupL0Reset.set(0)

class PVCtrls(object):

    def __init__(self, p, m, name=None, ip='0.0.0.0', xpm=None, stats=None, usTiming=None, handle=None, paddr=None, notify=True, db=None, cuInit=False, fidPrescale=200, fidPeriod=1400/1.3):
        global provider
        provider = p
        global lock
        lock     = m
        global _fidPrescale
        _fidPrescale = fidPrescale
        global _fidPeriod
        _fidPeriod = fidPeriod

        # Assign transmit link ID
        ip_comp = ip.split('.')
        xpm_num = name.rsplit(':',1)[1]
        v = 0x0ff00000 | ((int(xpm_num)&0xff)<<12)
        if ip_comp[3]!=0:
            v |= ((int(ip_comp[2])&0xf)<<8) | (((int(ip_comp[3])-100)&0xff)<<4)
        xpm.XpmApp.paddr.set(v)
        logging.info('Set PADDR to 0x{:08x}'.format(v))

        self._name  = name
        self._ip    = ip
        self._xpm   = xpm
        self._paddr = paddr
        self._db    = db
        self._seq   = None
        self._handle= handle

        init = None
        try:
            db_url, db_name, db_instrument, db_alias = db.split(',',4)
            logging.info('db {:}'.format(db))
            logging.info('url {:}  name {:}  instr {:}  alias {:}'.format(db_url,db_name,db_instrument,db_alias))
            logging.info('device {:}'.format(name))
            init = get_config_with_params(db_url, db_instrument, db_name, db_alias, name)
            logging.info('cfg {:}'.format(init))
        except:
            logging.warning('Caught exception reading configdb [{:}]'.format(db))

        self._links = []
        for i in range(24):
            self._links.append(LinkCtrls(name, xpm, i))

        app = xpm.XpmApp

        self._pv_amcDumpPLL = []

        self._cu    = CuGenCtrls(name+':XTPG', xpm, dbinit=init)

        self._group = GroupCtrls(name, app, stats, init=init)

        def _addPV(label,reg):
            pv = SharedPV(initial=NTScalar('I').wrap(0), 
                          handler=RegH(reg))
            provider.add(name+':'+label,pv)
            return pv

        self._pv_usRxReset      = _addPV('Us:RxReset'     ,xpm.UsTiming.RxReset)
        self._pv_usRxPllReset   = _addPV('Us:RxPllReset'  ,xpm.UsTiming.RxPllReset)
        self._pv_usRxCountReset = _addPV('Us:RxCountReset',xpm.UsTiming.RxCountReset)
        self._pv_cuRxReset      = _addPV('Cu:RxReset'     ,xpm.CuTiming.RxReset)
        self._pv_cuRxPllReset   = _addPV('Cu:RxPllReset'  ,xpm.CuTiming.RxPllReset)
        self._pv_cuRxCountReset = _addPV('Cu:RxCountReset',xpm.CuTiming.RxCountReset)

        self._pv_l0HoldReset = SharedPV(initial=NTScalar('I').wrap(0),
                                        handler=RegH(app.l0HoldReset,archive=False))
        provider.add(name+':L0HoldReset',self._pv_l0HoldReset)

        self._pv_seqReset   = SharedPV(initial=NTScalar('I').wrap(0),
#                                        handler=RegH(xpm.SeqEng_0.seqRestart,archive=False))
                                        handler=PVHandler(self.seqReset))
        provider.add(name+':SeqReset',self._pv_seqReset)

        self._pv_seqDone   = SharedPV(initial=NTScalar('I').wrap(0),
                                      handler=DefaultPVHandler())
        provider.add(name+':SeqDone',self._pv_seqDone)

        if usTiming:
            self._usLinkUp = usTiming._linkUpdate
            usTiming._linkUpdate = self.usLinkUp
        else:
            self._usLinkUp = None

        if notify:
            self._thread = threading.Thread(target=self.notify)
            self._thread.start()

        logging.info('monStreamPeriod {}'.format(app.monStreamPeriod.get()))
        app.monStreamPeriod.set(104166667)
        app.monStreamEnable.set(1)

    def usLinkUp(self):
        if self._usLinkUp is not None:
            self._usLinkUp()
        if self._seq:
            for s in self._seq:
                s.refresh()

    def seqReset(self,pv,val):
        self._xpm.SeqEng_0.seqRestart.set(val)
        if self._seq:
            for s in self._seq:
                if (1<<s._eng._id)&val:
                    s._eng.resetDone()

    def update(self,cycle):
        #  The following section will throw an exception if the CuInput PV is not set properly
        if cycle < 10:
            logging.info('pvseq in %d'%(10-cycle))
        elif cycle == 10:
            self._seq_codes_pv  = addPVT(self._name+':SEQCODES', seqCodes)
            self._seq_codes_val = toDict(seqCodes)
            self._seq = [PVSeq(provider, f'{self._name}:SEQENG:{i}', self._ip, Engine(i, self._xpm.SeqEng_0), self._seq_codes_val) for i in range(NCODES//4)]

        global countdn
        # check for config save
        if countdn > 0:
            countdn -= 1
            if countdn == 0 and self._db:
                # save config
                logging.info('Updating {}'.format(self._db))
                db_url, db_name, db_instrument, db_alias = self._db.split(',',4)
                mycdb = cdb.configdb(db_url, db_instrument, True, db_name, user=db_instrument+'opr')
                mycdb.add_device_config('xpm')

                top = cdict()
                top.setInfo('xpm', self._name, None, 'serial1234', 'No comment')
                top.setAlg('config', [0,0,0])

                lock.acquire()
                top.set('XTPG.CuDelay'   , self._xpm.CuGenerator.cuDelay.get()       , 'UINT32')
                top.set('XTPG.CuBeamCode', self._xpm.CuGenerator.cuBeamCode.get()    , 'UINT8')
                top.set('XTPG.CuInput'   , self._xpm.AxiSy56040.OutputConfig[0].get(), 'UINT8')
                v = []
                for i in range(8):
                    v.append( self._xpm.XpmApp.l0Delay(i) )
                top.set('PART.L0Delay', v, 'UINT32')
                lock.release()

                if not db_alias in mycdb.get_aliases():
                    mycdb.add_alias(db_alias)

                try:
                    mycdb.modify_device(db_alias, top)
                except:
                    pass
                
    def notify(self):
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.connect((self._ip,8197))
        
        msg = b'\x00\x00\x00\x00'
        client.send(msg)
        
        while True:
            msg = client.recv(4096)
            self.handle(msg)

    def handle(self,msg):
            s = struct.Struct('H')
            siter = s.iter_unpack(msg)
            src = next(siter)[0]

            if src==0:   # sequence notify message
                mask = next(siter)[0]
                i=0
                while mask!=0:
                    if mask&1:
                        addr = next(siter)[0]
                        logging.info('addr[{}] {:x}'.format(i,addr))
                        if i<1:
                            pvUpdate(self._pv_seqDone,1)
                    i += 1
                    mask = mask>>1
            elif src==1: # step end message
                group = next(siter)[0]
                self._group._groups[group].stepDone(True)
            elif src==2: # stats message
                if self._handle:
                    offset = self._handle(msg)
                    # seqcodes
                    if self._seq:
                        w = struct.unpack_from(f'<{NCODES}L',msg,offset) #seqCount
                        offset += NCODES*4

                        value = self._seq_codes_pv.current()
                        self._seq_codes_val['Rate'] = w
                        self._seq_codes_val['Description'] = value['value']['Description']
                        value['value'] = self._seq_codes_val
                        value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
                        self._seq_codes_pv.post(value)
                    else:
                        offset += NCODES*4

                    # paddr
                    if True:
                        v = struct.unpack_from(f'<I',msg,offset)[0] # pAddr
                        offset += 4
                        paddr = self._paddr.current()['value']
                        if v!=paddr:
                            logging.warning(f'Update paddr {paddr} {v}')
                            pvUpdate(self._paddr,v)

                    # seqInvalid
                    if self._seq:
                        u = struct.unpack_from(f'<B',msg,offset)[0] # seqInvalid
                        offset += 1
                        for i in range(NCODES//4):
                            if u&(1<<i):
                                logging.warning(f'Seq {i} is invalid.  Resetting.')
                                self._seq[i]._eng.reset()
                    else:
                        offset += 1
                                
                    # check for a dropped "step done"
                    for i in range(8):
                        g = self._group._groups[i]
                        if g._pv_StepGroups.current()['value']!=0:
                            end  = g._pv_StepEnd.current ()['value']
                            nL0  = g._stats._pv_numL0Acc.current()['value']
                            done = g._pv_StepDone.current()['value']
                            #logging.info(f'Check group {i}: end {end}  nL0 {nL0}  done {done}')
                            if ((end==nL0) and done==0):
                                logging.warning(f'Recover stepDone for group {i} events {end}')
                                g.stepDone(True)
