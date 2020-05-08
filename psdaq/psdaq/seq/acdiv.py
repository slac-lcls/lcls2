from psdaq.seq.seq import *

rates = [360,180,120, 90, 72, 60, 45, 40, 36, 30, 24, 20, 18, 15, 12, 10]
div   = [  1,  2,  3,  4,  5,  6,  8,  9, 10, 12, 15, 18, 20, 24, 30, 36]
sync_marker = 4  # 1Hz AC
    
instrset = []
#  Insert global sync instruction for timeslot 1
instrset.append(ACRateSync(timeslotm=(1<<0),marker=sync_marker,occ=1))

for i in range(360):
    b = 0
    for j in range(16):
        if (i%div[j])==0:
            b |= (1<<j)
    instrset.append(ControlRequest(b))
    instrset.append(ACRateSync(timeslotm=0x3f,marker=0,occ=1))
instrset.append(Branch.unconditional(line=1))

descset = []
for j in range(16):
    descset.append('{:}Hz'.format(rates[j]))

title = 'ACDIV'
