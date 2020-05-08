from psdaq.seq.seq import *

sync_marker = 4  # 1Hz AC
    
instrset = []
#  Insert global sync instruction for timeslot 1
instrset.append(ACRateSync(timeslotm=(1<<0),marker=sync_marker,occ=1))

instrset.append(ControlRequest(1))
instrset.append(ACRateSync(timeslotm=(1<<0)|(1<<2)|(1<<4),marker=0,occ=1))
instrset.append(Branch.unconditional(line=1))

descset = ['180Hz']

title = 'AC180'
