from psana import DataSource

def filter(evt):
    return True

# in future may need beginjob/beginrun/updateconfig/endrun/endjob callbacks in addition 
# to analyze() 
        
def analyze(evt): # Use detector interface here and in filter
    pass

ds = DataSource('exp=xpptut13:run=1')
ds.start(analyze, filter=filter, lbatch=2)

######## end of user.py
