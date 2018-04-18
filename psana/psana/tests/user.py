# Todo
# - Use detector interface in eventCode

from psana import DataSource

def filter(evt):
    return True

# Usecase#1 : two iterators
ds = DataSource('exp=xpptut13:run=1', filter=filter)
#beginJobCode
for run in ds.runs():
    #beginRunCode
    for evt in run.events():
        #eventCode
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
