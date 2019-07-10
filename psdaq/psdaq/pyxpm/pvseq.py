import time

from psdaq.cas.seq import *
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
                    seq.append(CheckPoint(0))
                elif instr == ControlRequest.opcode:
                    seq.append(ControlRequest(args[1]))
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
            for i in self._seq:
                if i.opcode == Branch.opcode:
                    jumpto = i.address()
                    if jumpto > len(self._seq):
                        rval = -3
                    elif jumpto >= 0:
                        jaddr = 0
                        for j,seq in enumerate(self._seq):
                            if j==jumpto:
                                break
                            jaddr += _nwords(seq)
                        self._ram[addr].set(i._word(jaddr+best_ram))
                        addr += 1
                else:
                    self._ram[addr].set(i._word())
                    addr += 1

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

    def dump(self):
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
                    print('[{:08x}] {:08x}'.format(key+j,self._ram[key+j].get()))

class PVSeq(object):
    def __init__(self, provider, name, ip, engine):
        self._eng = engine
        self._seq = []

        def addPV(label,ctype='I',init=0):
            pv = SharedPV(initial=NTScalar(ctype).wrap(init), 
                          handler=DefaultPVHandler())
            provider.add(name+':'+label,pv)
            return pv

        self._pv_DescInstrs    = addPV('DESCINSTRS','s','')
        self._pv_InstrCnt      = addPV('INSTRCNT')
        self._pv_SeqIdx        = addPV('SEQIDX'    ,'aI',[0]*NSubSeq)
        self._pv_SeqDesc       = addPV('SEQDESC'   ,'as',['']*NSubSeq)
        self._pv_Seq00Idx      = addPV('SEQ00IDX')
        self._pv_Seq00Desc     = addPV('SEQ00DESC' ,'s','')
        self._pv_Seq00BDesc    = addPV('SEQ00BDESC','as',['']*NSubSeq)
        self._pv_RmvIdx        = addPV('RMVIDX')
        self._pv_RunIdx        = addPV('RUNIDX')
        self._pv_Running       = addPV('RUNNING')

        def addPV(label,ctype,init,cmd):
            pv = SharedPV(initial=NTScalar(ctype).wrap(init), 
                          handler=PVHandler(cmd))
            provider.add(name+':'+label,pv)
            return pv

        self._pv_Instrs        = addPV('INSTRS'    ,'aI',[0]*16384, self.instrs)
        self._pv_RmvSeq        = addPV('RMVSEQ'    , 'I',        0, self.rmvseq)
        self._pv_Ins           = addPV('INS'       , 'I',        0, self.ins)
        self._pv_SchedReset    = addPV('SCHEDRESET', 'I',        0, self.schedReset)
        self._pv_ForceReset    = addPV('FORCERESET', 'I',        0, self.forceReset)

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
        if val:
            idx = self._pv_RunIdx.current()['value']
            print('Scheduling index {}',idx)
            pvUpdate(self._pv_Running,1 if idx>1 else 0)
            self._eng.enable(True)
            self._eng.setAddress(idx,0,1)
            self._eng.reset()

    def forceReset(self, pv, val):
        if val:
            idx = self._pv_RunIdx.current()['value']
            print('Starting index {}',idx)
            pvUpdate(self._pv_Running,1 if idx>1 else 0)
            self._eng.enable(True)
            self._eng.setAddress(idx,0,0)
            self._eng.reset()

    def checkPoint(self,addr):
        pvUpdate(self._pv_Running,0)

