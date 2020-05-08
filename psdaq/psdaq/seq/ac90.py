from psdaq.seq.seq import *

sync_marker = 4  # 1Hz AC
    
instrset = []
#  Insert global sync instruction for timeslot 1
instrset.append(ACRateSync(timeslotm=(1<<1),marker=sync_marker,occ=1))

instrset.append(ControlRequest(1))
instrset.append(ACRateSync(timeslotm=(1<<5),marker=0,occ=1))
instrset.append(ControlRequest(1))
instrset.append(ACRateSync(timeslotm=(1<<3),marker=0,occ=1))
instrset.append(ControlRequest(1))
instrset.append(ACRateSync(timeslotm=(1<<1),marker=0,occ=1))
instrset.append(Branch.unconditional(line=1))

descset = ['90Hz']

title = 'AC90'
