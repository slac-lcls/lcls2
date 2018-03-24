# Todo
# - In future may need beginjob/beginrun/updateconfig/endrun/endjob callbacks in addition 
#   to analyze()
# - Use detector interface in callbacks

from psana import DataSource

def filter(evt):
    return True
        
def analyze(evt):
    pass 

ds = DataSource('exp=xpptut13:run=1')
ds.start(analyze, filter=filter)

