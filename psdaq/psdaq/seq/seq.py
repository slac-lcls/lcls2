#
#  This code needs validation checks:
#  1.  arguments do not exceed the bit depth of the implementation
#  2.  consistent use of the branch counters (same counter isn't used within a nested loop)
#
from psdaq.configdb.tsdef import *
import math

verbose = False
#verbose = True

def factor(n):
    if n <= Instruction.maxocc:
        return (n)

    if n > Instruction.maxocc * Instruction.maxocc:
        raise ValueError('factor failed: argument too large')

    primes = []
    rem = n
    for i in range(2,int(math.sqrt(n)+1)):
        while rem%i == 0:
            primes.append(i)
            rem = rem//i
        if rem == 1:
            break

    print(f'primes of {n} are {primes}')

    # Can we make two factors each less than Instruction.maxoc

class Instruction(object):

    maxocc = 0xfff

    def __init__(self, args):
        self.args = args

    def encoding(self):
        args = [0]*7
        args[0] = len(self.args)-1
        args[1:len(self.args)+1] = self.args
        return args

    def __str__(self):
        return self.print_()

    # In subclasses that need address relocation
    #def address(self): 
    #def _word(self,a):

class FixedRateSync(Instruction):

    opcode = 0

    def __init__(self, marker, occ):
        if occ > Instruction.maxocc:
            raise ValueError('FixedRateSync called with occ={}'.format(occ))
        if marker in FixedIntvsDict:
            mk     = marker
            marker = FixedIntvsDict[mk]['marker']
            self.intv = FixedIntvsDict[mk]['intv']
        else:
            self.intv = FixedIntvs[marker]
        super(FixedRateSync, self).__init__( (self.opcode, marker, occ) )

    def _word(self):
        return int((2<<29) | ((self.args[1]&0xf)<<16) | (self.args[2]&Instruction.maxocc))
    
    def print_(self):
        return 'FixedRateSync({}) # occ({})'.format(fixedRates[self.args[1]],self.args[2])

    def execute(self,engine):
        intv = self.intv
        engine.instr += 1
        step = intv*self.args[2]-(engine.frame%intv)
        if step>0:
            engine.frame  += step
            engine.request = 0
        engine.modes |= 1

class ACRateSync(Instruction):

    opcode = 1

    def __init__(self, timeslotm, marker, occ):
        if occ > Instruction.maxocc:
            raise ValueError('ACRateSync called with occ={}'.format(occ))
        if timeslotm > 0x3f:
            raise ValueError('ACRateSync called with timeslotm={}'.format(timeslotm))
        if marker in ACIntvsDict:
            mk     = marker
            marker = ACIntvsDict[mk]['marker']
            self.intv = ACIntvsDict[mk]['intv']
        else:
            self.intv = ACIntvs[marker]
        super(ACRateSync, self).__init__( (self.opcode, timeslotm, marker, occ) )

    def _word(self):
        return int((3<<29) | ((self.args[1]&0x3f)<<23) | ((self.args[2]&0xf)<<16) | (self.args[3]&Instruction.maxocc))

    def print_(self):
        return 'ACRateSync({}/0x{:x}) # occ({})'.format(acRates[self.args[2]],self.args[1],self.args[3])
    
    def execute(self,engine):
        intv = self.intv
        engine.instr += 1
        mask = self.args[1]&0x3f
#        print('ACRateSync: args {:}  mask {:x}  intv {:}'.format(self.args,mask,intv))
        for i in range(self.args[3]):
            while True:
                engine.acframe += 1
                ts = engine.acframe % 6
#                print('  frame {:}  ts {:}'.format(engine.acframe,ts))
                if ((1<<ts)&mask)!=0 and (int(engine.acframe/6)%intv)==0:
                    break

        engine.request = 0
        engine.modes  |= 2

class Branch(Instruction):

    opcode = 2

    def __init__(self, args):
        if len(args)>2:
            if args[2] > 0x3:
                raise ValueError('Branch called with ctr={}'.format(args[2]))
            if args[3] > Instruction.maxocc:
                raise ValueError('Branch called with occ={}'.format(args[3]))
        super(Branch, self).__init__(args)

    def _word(self, a):
        w = a & 0x7ff
        if len(self.args)>2:
            w = ((self.args[2]&0x3)<<27) | (1<<24) | ((self.args[3]&Instruction.maxocc)<<12) | w
        return int(w)

    @classmethod
    def unconditional(cls, line):
        return cls((cls.opcode, line))

    @classmethod
    def conditional(cls, line, counter, value):
        return cls((cls.opcode, line, counter, value))

    def address(self):
        return self.args[1]

    def print_(self):
        if len(self.args)==2:
            return 'Branch unconditional to line {}'.format(self.args[1])
        else:
            return 'Branch to line {} until ctr{}={}'.format(self.args[1],self.args[2],self.args[3])

    def execute(self,engine):
        if len(self.args)==2:
            if engine.instr==self.args[1]:  # branch to self
                engine.done = True
            engine.instr = self.args[1]
        else:
            if engine.ccnt[self.args[2]]==self.args[3]:
                engine.instr += 1
                engine.ccnt[self.args[2]] = 0
            else:
                engine.instr = self.args[1]
                engine.ccnt[self.args[2]] += 1
    
class CheckPoint(Instruction):

    opcode = 3
    
    def __init__(self):
        super(CheckPoint, self).__init__((self.opcode,))

    def _word(self):
        return int((1<<29))

    def print_(self):
        return 'CheckPoint'

    def execute(self,engine):
        engine.instr += 1

class BeamRequest(Instruction):

    opcode = 4
    
    def __init__(self, charge):
        super(BeamRequest, self).__init__((self.opcode, charge))

    def _word(self):
        return int((4<<29) | self.args[1])

    def print_(self):
        return 'BeamRequest charge {}'.format(self.args[1])

    def execute(self,engine):
        engine.request = (self.args[1]<<16) | 1
        engine.instr += 1

class ControlRequest(Instruction):

    opcode = 5
    
    def __init__(self, word):
        if isinstance(word,list):
            v = 0
            for w in word:
                v |= (1<<w)
        else:
            v = word
        super(ControlRequest, self).__init__((self.opcode, v))
 
    def _word(self):
        return int((4<<29) | self.args[1])

    def print_(self):
        codes = []
        w = self.args[1]
        code = 0
        while w:
            if w&1:
                codes.append(code)
            w >>= 1
            code += 1

        return f'ControlRequest word 0x{self.args[1]:x} {codes}'

    def execute(self,engine):
        engine.request = self.args[1]
        engine.instr += 1

class Call(Instruction):

    opcode = 6
    
    def __init__(self, line):
        super(Call, self).__init__((self.opcode,line))

    def _word(self,a):
        return int((5<<29) | (a&0x7ff))

    def address(self):
        return self.args[1]

    def print_(self):
        return f'Call 0x{self.args[1]:x}'

    def execute(self,engine):
        engine.returnaddr = engine.instr+1
        engine.instr = self.args[1]

class Return(Instruction):

    opcode = 7

    def __init__(self):
        super(Return, self).__init__((self.opcode,))
 
    def _word(self):
        return int((5<<29) | (1<<12))

    def print_(self):
        return f'Return'

    def execute(self,engine):
        if engine.returnaddr is None:
            raise ValueError(f'engine.returnaddr is None')
        engine.instr = engine.returnaddr
        engine.returnaddr = None

def decodeInstr(w):
    idw = w>>29
    instr = Instruction([])
    if idw == 0:  # Branch
        if w&(1<<24):
            instr = Branch.conditional(line=w&0x7ff,counter=(w>>27)&3,value=(w>>12)&Instruction.maxocc)
        else:
            instr = Branch.unconditional(line=w&0x7ff)
    elif idw == 1: # Checkpoint
        instr = CheckPoint()
    elif idw == 2: # FixedRateSync
        instr = FixedRateSync(marker=(w>>16)&0xf,occ=w&Instruction.maxocc)
    elif idw == 3: # ACRateSync
        instr = ACRateSync(timeslotm=(w>>23)&0x3f,marker=(w>>16)&0xf,occ=w&Instruction.maxocc)
    elif idw == 4: # Request (assume ControlRequest)
        instr = ControlRequest(word = w&0xffff)
    elif idw == 5: # Call/Return
        if (w&(1<<12)):
            instr = Subroutine.return_()
        else:
            instr = Subroutine.call(w&0xfff)
    return instr

#  validate the conditional counters in a list of instructions
def validate(filename):
    config = {'title':'TITLE', 'descset':None, 'instrset':None, 'seqcodes':None, 'repeat':False}
    seq = 'from psdaq.seq.seq import *\n'
    seq += open(filename).read()
    exec(compile(seq, filename, 'exec'), {}, config)
    l = config['instrset']

#    for i,ins in enumerate(l):
#        print(f'{i}: {ins}')

    #  accumulate the branch statement source and targets
    d = {cc:[] for cc in range(4)}
    for line,instr in enumerate(l):
        if instr.args[0]==Branch.opcode and len(instr.args)>2:
            cc   = instr.args[2]
            addr = instr.args[1]
            d[cc].append([addr,line])

#    for i,dd in d.items():
#        print(f'd[{i}] = {dd}')

    #  check none of them overlap for a given conditional counter
    for cc in range(4):
        for r in d[cc]:
            addr = r[0]
            for s in d[cc]:
                if addr>s[0] and addr<s[1]:
                    raise ValueError(f'{filename}: CC {cc} found in overlapping loops {r} {s}')

    #  don't know how to validate call/return matches


#  Translate instruction addresses
def relocate(instrset,target,source=0):
    words = []
    for i in instrset:
        if hasattr(i,'address'):
            jumpto = i.address()
            if jumpto > len(instrset)+source:
                return None
            elif jumpto >= source:
                words.append(i._word(jumpto+target))
            else:
                return None
        else:
            words.append(i._word())

    for i,ins in enumerate(instrset):
        print(f'{i}: {ins}')

    for i,w in enumerate(words):
        print(f'{i}: {w:x}')

    return words
