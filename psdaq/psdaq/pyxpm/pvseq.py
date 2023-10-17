import time

from psdaq.seq.seq import *
from psdaq.pyxpm.pvhandler import *
from p4p.nt import NTScalar
from p4p.server.thread import SharedPV

verbose = True

NSubSeq = 64

def _nwords(instr):
    return 1

class SeqCache(object):
    def __init__(self, index, sz, instr):
        self.index = index
        self.size  = sz
        self.instr = instr

class Engine(object):

    def __init__(self, id, reg):
        self._id   = id
        self._reg  = reg
        self._jump = reg.find(name='SeqJump_%d'%id)[0]
        self._ram  = reg.find(name='SeqMem_%d'%id)[0].mem
        self._caches  = {}  # instruction sequences committed to device

        #  Track sequences that can be reset on 1Hz marker repeatedly
        #  _refresh = 0 : only reset explicitly
        #  _refresh = 1 : wait for explicit reset, then go to 2
        #  _refresh = 2 : reset on linkUp
        self._refresh = 0   

        self._indices = 0
        self.dump()

        a = 0
        self._caches[a]                           = SeqCache(0,3,[FixedRateSync(5,1),FixedRateSync(5,1),Branch.unconditional(line=0)])
        self._ram   [a  ].set(FixedRateSync(5,1)._word())
        self._ram   [a+1].set(FixedRateSync(5,1)._word())
        self._ram   [a+2].set(Branch.unconditional(line=0)._word(a))
        a = (1<<reg.seqAddrLen.get())-1
        self._caches[a] = SeqCache(1,1,[Branch.unconditional(line=0)])
        self._ram   [a].set(Branch.unconditional(line=a)._word(a))
        self._indices = 3   # bit mask of committed sequences
        self._seq     = []  # instruction sequence to be committed

    def cacheSeq(self,val):
        seq = []
        try:
            #  Reconstitute the list of instructions
            iiter = iter(val)
            ninstr = next(iiter)
            for i in range(ninstr):        
                nargs = next(iiter)
                args  = [ next(iiter) for i in range(6) ]
                instr = args[0]
                if instr == FixedRateSync.opcode:
                    seq.append(FixedRateSync(args[1],args[2]))
                elif instr == ACRateSync.opcode:
                    seq.append(ACRateSync(args[1],args[2],args[3]))
                elif instr == Branch.opcode:
                    if nargs == 1:
                        seq.append(Branch.unconditional(args[1]))
                    else:
                        seq.append(Branch.conditional(args[1],args[2],args[3]))
                elif instr == CheckPoint.opcode:
                    seq.append(CheckPoint())
                elif instr == ControlRequest.opcode:
                    seq.append(ControlRequest(args[1]))
                elif instr == Call.opcode:
                    seq.append(Call(args[1]))
                elif instr == Return.opcode:
                    seq.append(Return())

        except StopIteration:
            pass
        self._seq = seq
        return len(seq)

    def insertSeq(self):
        rval = 0
        aindex = -3

        while True:
            # Validate sequence (skip)
            # Calculate memory needed
            nwords = 0
            for i in self._seq:
                nwords = nwords + _nwords(i)

            # Find memory range (just) large enough
            best_ram = 0
            if True:
                addr = 0
                none_found = 1<<self._reg.seqAddrLen.get()
                best_size = none_found
                for key,cache in self._caches.items():
                    isize = key-addr
                    if verbose:
                        print('Found memblock {:x}:{:x} [{:x}]'.format(addr,key,isize))
                    if isize==nwords:
                        best_size = isize
                        best_ram  = addr
                        break
                    elif isize > nwords and isize < best_size:
                        best_size = isize
                        best_ram  = addr
                    addr = key+cache.size
                if best_size == none_found:
                    print('BRAM space unavailable')
                    rval = -1
                    break
                if verbose:
                    print('Using memblock {:x}:{:x}  [{:x}]'.format(best_ram,best_ram+nwords,nwords))
            if rval:
                break
            if self._indices == -1:
                rval = -2
                break

            for i in range(NSubSeq):
                if (self._indices & (1<<i))==0:
                    self._indices = self._indices | (1<<i)
                    aindex = i
                    break

            print('Caching seq {} of size {}'.format(aindex,nwords))
            self._caches[best_ram] = SeqCache(aindex,nwords,self._seq)

            #  Translate addresses
            addr = best_ram
            words = relocate(self._seq,best_ram)
            if words is None:
                rval = -3
            else:
                for i,w in enumerate(words):
                    self._ram[best_ram+i].set(w)

            print('Translated addresses rval = {}'.format(rval))

            if rval:
                self.removeSeq(aindex)
            break

        if rval==0:
            rval = aindex
        return rval

    def removeSeq(self, index):
        if (self._indices & (1<<index))==0:
            return -1
        self._indices = self._indices & ~(1<<index)

        # Lookup sequence
        ram  = self._reg.SeqMem_0.mem
        for key,seq in self._caches.items():
            if seq.index == index:
                self._ram[key].set(key)
                del self._caches[key]
                return 0
        return -2

    def setAddress(self, seq, start, sync):
        a = -1
        for key,entry in self._caches.items():
            if entry.index == seq:
                a = key
                for i in range(start):
                    a += _nwords(entry.instr)
                break

        if a>=0:
            self._jump.setManStart(a,0)
            self._jump.setManSync (sync)
            print('sequence started at address 0x{:x}'.format(a))
        else:
            print('sequence {} failed to start'.format(seq))

    def enable(self,e):
        v = self._reg.seqEn.get()
        if e:
            v = v | (1<<self._id)
        else:
            v = v & ~(1<<self._id)
        self._reg.seqEn.set(v)

    def reset(self):
        v = 1<<self._id
        self._reg.seqRestart.set(v)
        self.resetDone()

    def resetDone(self):
        if self._refresh==1:
            self._refresh=2

    def dump(self):
        print('seqAddrLen %d'%self._reg.seqAddrLen.get())
        print('ctlSeq     %d'%self._reg.ctlSeq.get())
        print('xpmSeq     %d'%self._reg.xpmSeq.get())

        v = self._jump.reg[15].get()
        print('Sync [{:04x}]  Start [{:04x}]  Enable [{:08x}]'.format(v>>16,v&0xffff,self._reg.seqEn.get()))
        state = self._reg.find(name='SeqState_%d'%self._id)[0]
        cond  = state.find(name='cntCond')
        print("Req {:08x}  Inv {:08x}  Addr {:08x}  Cond {:02x}{:02x}{:02x}{:02x}"
              .format(state.find(name='cntReq')    [0].get(),
                      state.find(name='cntInv')    [0].get(),
                      state.find(name='currAddr')  [0].get(),
                      cond[0].get(),
                      cond[1].get(),
                      cond[2].get(),
                      cond[3].get()))
        for i in range(NSubSeq):
            if (self._indices & (1<<i)):
                print('Sequence %d'%i)
                self.dumpSequence(i)

    def dumpSequence(self,i):
        for key,entry in self._caches.items():
            if entry.index == i:
                for j in range(entry.size):
                    ram = self._ram[key+j].get()
                    print('[{:08x}] {:08x} {:}'.format(key+j,ram,decodeInstr(ram)))


class PVSeq(object):
    def __init__(self, provider, name, ip, engine, pv_enabled):
        self._eng = engine
        self._seq = []
        self._pv_enabled = pv_enabled

        def _addPV(label,ctype='I',init=0):
            pv = SharedPV(initial=NTScalar(ctype).wrap(init), 
                          handler=DefaultPVHandler())
            provider.add(name+':'+label,pv)
            print(name+':'+label)
            return pv

        self._pv_DescInstrs    = _addPV('DESCINSTRS','s','')
        self._pv_InstrCnt      = _addPV('INSTRCNT')
        self._pv_SeqIdx        = _addPV('SEQIDX'    ,'aI',[0]*NSubSeq)
        self._pv_SeqDesc       = _addPV('SEQDESC'   ,'as',['']*NSubSeq)
        self._pv_Seq00Idx      = _addPV('SEQ00IDX')
        self._pv_Seq00Desc     = _addPV('SEQ00DESC' ,'s','')
        self._pv_Seq00BDesc    = _addPV('SEQ00BDESC','as',['']*NSubSeq)
        self._pv_RmvIdx        = _addPV('RMVIDX')
        self._pv_RunIdx        = _addPV('RUNIDX')
        self._pv_Running       = _addPV('RUNNING')

        def _addPV(label,ctype,init,cmd):
            pv = SharedPV(initial=NTScalar(ctype).wrap(init), 
                          handler=PVHandler(cmd))
            provider.add(name+':'+label,pv)
            return pv

        self._pv_Instrs        = _addPV('INSTRS'    ,'aI',[0]*16384, self.instrs)
        self._pv_RmvSeq        = _addPV('RMVSEQ'    , 'I',        0, self.rmvseq)
        self._pv_Ins           = _addPV('INS'       , 'I',        0, self.ins)
        self._pv_SchedReset    = _addPV('SCHEDRESET', 'I',        0, self.schedReset)
        self._pv_ForceReset    = _addPV('FORCERESET', 'I',        0, self.forceReset)
        self._pv_Enable        = _addPV('ENABLE'    , 'I',        0, self.enable)
        self._pv_Dump          = _addPV('DUMP'      , 'I',        0, self.dump)

        if engine._reg.seqEn.get()&(1<<engine._id):
            self.enable(None,1)

#        self.updateInstr()

    def instrs(self, pv, val):
        pvUpdate(self._pv_InstrCnt,self._eng.cacheSeq(val))

    def rmvseq(self, pv, pval):
        val = self._pv_RmvIdx.current()['value']
        print('rmvseq index %d'%val)
        if val > 1 and val < NSubSeq:
            self._eng.removeSeq(val)
            pvUpdate(self._pv_Seq00Idx,0)

    def ins(self, pv, val):
        if val:
            rval = self._eng.insertSeq()
            pvUpdate(self._pv_Seq00Idx,rval)

    def schedReset(self, pv, val):
        if val>0:
            idx = self._pv_RunIdx.current()['value']
            print(f'Scheduling index {idx}')
            pvUpdate(self._pv_Running,1 if idx>1 else 0)
            self.enable(None,1)
            self._eng.setAddress(idx,0,0)  # syncs start to marker 0 (1Hz)
            self._eng._refresh = 0
            if val==1:
                self._eng.reset()
            elif val==2:
                pass  # No reset
            elif val==3:
                self._eng._refresh = 1
                self._eng.reset()
            elif val==4:
                self._eng._refresh = 1

    def forceReset(self, pv, val):
        if val:
            idx = self._pv_RunIdx.current()['value']
            print(f'Starting index {idx}')
            pvUpdate(self._pv_Running,1 if idx>1 else 0)
            self.enable(None,1)
            self._eng.setAddress(idx,0,6)  # syncs start to marker 6 (MHz)
            self._eng.reset()

    def enable(self, pv, val):
        id = self._eng._id
        self._pv_enabled['Enabled'][4*id:4*id+4] = [(val!=0)]*4
        self._eng.enable(val)

    def checkPoint(self,addr):
        pvUpdate(self._pv_Running,0)

    def dump(self, pv, val):
        self._eng.dump()

    def refresh(self):
        if self._eng._refresh > 1:
            self._eng.reset()
