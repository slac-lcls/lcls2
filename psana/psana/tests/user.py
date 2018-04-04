# Todo
# - Use detector interface in eventCode

from psana import DataSource

def filter(evt):
    return True

ds = DataSource('exp=xpptut13:run=1', filter=filter)
#beginJobCode
for run in ds.runs():
    #beginRunCode
    for evt in run.events():
        #eventCode
        pass 
    #endRunCode
#endJobCode

