# NOTE:
# To test on 'real' bigdata: 
# xtc_dir = "/reg/d/psdm/xpp/xpptut15/scratch/mona"
# >bsub -n 64 -q psfehq -o log.txt mpirun python user.py
#
# Todo
# - Use detector interface in eventCode
import os
from psana import DataSource

def filter(evt):
    return True

# Usecase#1 : two iterators
xtc_dir = os.path.join(os.getcwd(),'.tmp')
ds = DataSource('exp=xpptut13:dir=%s'%(xtc_dir), filter=filter)

#beginJobCode
for run in ds.runs():
    #beginRunCode
    for evt in run.events():
        #eventCode
        for d in evt:
            pass
    #endRunCode
#endJobCode

# Usecase#2: one iterator
for evt in ds.events():
    pass

# Usecase#3: looping through configs
for run in ds.runs():
    for configUpdate in run.configUpdates():
        for config in configUpdate.events():
            pass
