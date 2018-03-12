from psana import DataSource

def filter(evt):
    return True

# in future may need beginjob/beginrun/updateconfig/endrun/endjob callbacks in addition 
# to analyze() 
        
def analyze(evt): # Use detector interface here and in filter
    print("Event")
    for dgram in evt:
        for var_name in sorted(vars(dgram)):
            val=getattr(dgram, var_name)
            print("  %s: %s" % (var_name, type(val)))

ds = DataSource('exp=xpptut13:run=1')
ds.start(analyze, filter=filter)

######## end of user.py
