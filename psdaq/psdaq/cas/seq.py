import time

fixedRates = ['929kHz','71.4kHz','10.2kHz','1.02kHz','102Hz','10.2Hz','1.02Hz']
acRates    = ['60Hz','30Hz','10Hz','5Hz','1Hz','0.5Hz']

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
        return 'FixedRateSync(%s) # occ(%d)'%(fixedRates[self.args[1]],self.args[2])
    

class ACRateSync(Instruction):

    opcode = 1

    def __init__(self, timeslotm, marker, occ):
        super(ACRateSync, self).__init__( (self.opcode, timeslotm, marker, occ) )

    def _word(self):
        return int((3<<29) | ((self.args[1]&0x3f)<<23) | ((self.args[2]&0xf)<<16) | (self.args[3]&0xfff))

    def print_(self):
        return 'ACRateSync(%s/0x%x) # occ(%d)'%(acRates[self.args[2]],self.args[1],self.args[3])
    

class Branch(Instruction):

    opcode = 2

    def __init__(self, args):
        super(Branch, self).__init__((Branch.opcode,)+args)

    def _word(self, a):
        w = a & 0x7ff
        if len(self.args)>2:
            w = ((self.args[2]&0x3)<<27) | (1<<24) | ((self.args[3]&0xff)<<16) | w
        return int(w)

    @classmethod
    def unconditional(cls, line):
        return cls((line,))

    @classmethod
    def conditional(cls, line, counter, value):
        return cls((line, counter, value))

    def address(self):
        return self.args[1]

    def print_(self):
        if len(self.args)==2:
            return 'Branch unconditional to line %d'%self.args[1]
        else:
            return 'Branch to line %d until ctr%d=%d'%(self.args[1:])
    
class BeamRequest(Instruction):

    opcode = 4
    
    def __init__(self, charge):
        super(BeamRequest, self).__init__((self.opcode, charge))

    def _word(self):
        return int((4<<29) | self.args[1])

    def print_(self):
        return 'BeamRequest charge %d'%self.args[1]


class ControlRequest(Instruction):

    opcode = 5
    
    def __init__(self, word):
        super(ControlRequest, self).__init__((self.opcode, word))

    def _word(self):
        return int((4<<29) | self.args[1])

    def print_(self):
        return 'ControlRequest word 0x%x'%self.args[1]

class CheckPoint(Instruction):

    opcode = 3
    
    def __init__(self, word):
        super(CheckPoint, self).__init__((self.opcode, word))

    def _word(self):
        return int(1<<29)

    def print_(self):
        return 'CheckPoint 0x%x'%self.args[1]

