from psdaq.seq.seq import *

sync_marker = 6
    
instrset = []
#  Insert global sync instruction (1Hz?)
instrset.append(FixedRateSync(marker=sync_marker,occ=1))

#  Setup a 90 pulse sequence to repeat 10000 times each second
for i in range(90):
    bits = 0
    for j in range(16):
        if i*(j+1)%90 < j+1:
            bits = bits | (1<<j)
    instrset.append(ControlRequest(bits))
    instrset.append(FixedRateSync(marker=0,occ=1))

instrset.append(Branch.conditional(line=1, counter=0, value=99))
instrset.append(Branch.conditional(line=1, counter=1, value=99))
instrset.append(Branch.unconditional(line=0))

descset = []
for j in range(16):
    descset.append('%d kHz'%((j+1)*10))

i=0
for instr in instrset:
    print( 'Put instruction(%d): '%i), 
    print( instr.print_())
    i += 1


title = 'LoopTest'
