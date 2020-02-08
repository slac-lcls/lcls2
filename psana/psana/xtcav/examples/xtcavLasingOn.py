import psana
from xtcav.LasingOnCharacterization import *
import numpy as np

run = '137'
experiment='amox23616'
maxshots = 50

XTCAVRetrieval=LasingOnCharacterization() 
data_source = psana.DataSource("exp=%s:run=%s:smd" % (experiment, run))
n_r=0  #Counter for the total number of xtcav images processed within the run 
for evt in data_source.events():
    if not XTCAVRetrieval.processEvent(evt):
        continue

    t, power = XTCAVRetrieval.xRayPower()  
    agreement = XTCAVRetrieval.reconstructionAgreement()
    pulse = XTCAVRetrieval.pulseDelay()
    print 'Agreement: %g%% Maximum power: %g GW Pulse Delay: %g ' %(agreement*100,np.amax(power), pulse[0])
    
    n_r += 1   

    if n_r>=maxshots: 
        break

