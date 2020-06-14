verbose = False
#verbose = True

fixedRates = ['929kHz','71.4kHz','10.2kHz','1.02kHz','102Hz','10.2Hz','1.02Hz']
acRates    = ['60Hz','30Hz','10Hz','5Hz','1Hz','0.5Hz']
FixedIntvs = [1, 13, 91, 910, 9100, 91000, 910000]
ACIntvs    = [1, 2, 6, 12, 60, 120]

class Instruction(object):

    def __init__(self, args):
        self.args = args

    def encoding(self):
        args = [0]*7
        args[0] = len(self.args)-1
        args[1:len(self.args)+1] = self.args
        return args

class FixedRateSync(Instruction):

    opcode = 0

    def __init__(self, marker, occ):
        super(FixedRateSync, self).__init__( (self.opcode, marker, occ) )

    def _word(self):
        return int((2<<29) | ((self.args[1]&0xf)<<16) | (self.args[2]&0xfff))
    
    def print_(self):
        return 'FixedRateSync({}) # occ({})'.format(fixedRates[self.args[1]],self.args[2])

    def execute(self,engine):
        intv = FixedIntvs[self.args[1]]
        engine.instr += 1
        step = intv*self.args[2]-(engine.frame%intv)
        if step>0:
            engine.frame  += step
            engine.request = 0
        engine.modes |= 1

class ACRateSync(Instruction):

    opcode = 1

    def __init__(self, timeslotm, marker, occ):
        super(ACRateSync, self).__init__( (self.opcode, timeslotm, marker, occ) )

    def _word(self):
        return int((3<<29) | ((self.args[1]&0x3f)<<23) | ((self.args[2]&0xf)<<16) | (self.args[3]&0xfff))

    def print_(self):
        return 'ACRateSync({}/0x{:x}) # occ({})'.format(acRates[self.args[2]],self.args[1],self.args[3])
    
    def execute(self,engine):
        intv = ACIntvs[self.args[2]]
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
        super(Branch, self).__init__(args)

    def _word(self, a):
        w = a & 0x7ff
        if len(self.args)>2:
            w = ((self.args[2]&0x3)<<27) | (1<<24) | ((self.args[3]&0xff)<<16) | w
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
        super(ControlRequest, self).__init__((self.opcode, word))
 
    def _word(self):
        return int((4<<29) | self.args[1])

    def print_(self):
        return 'ControlRequest word 0x{:x}'.format(self.args[1])

    def execute(self,engine):
        engine.request = self.args[1]
        engine.instr += 1

