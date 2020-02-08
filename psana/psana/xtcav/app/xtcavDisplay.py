#!/usr/bin/env python
from psana import *
from xtcav2 import Constants

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("experiment", help="psana experiment string (e.g. 'xppd7114')")
parser.add_argument("run", type=int, help="run number")
args = parser.parse_args()

ds = DataSource("exp=%s:run=%s:smd" % (args.experiment, str(args.run)))
xtcav = Detector(Constants.SRC, ds.env())
gdet = Detector(Constants.GAS_DETECTOR)

def getLasingOffShot(XTCAVRetrieval,expt):
    results=XTCAVRetrieval._pulse_characterization
    lor = XTCAVRetrieval._lasingoffreference
    ibunch = 0

    group = results.groupnum[ibunch]
    profs = lor.averaged_profiles

    ds_lasingoff = DataSource('exp=%s:run=%s:idx'%(expt,lor.parameters.run))
    run = ds_lasingoff.runs().next()
    times = run.times()
    time = profs.eventTime[ibunch][group]
    fid = profs.eventFid[ibunch][group]
    et = EventTime(int(time),int(fid))
    evt_lasingoff = run.event(et)
    xtcav_lasingoff = Detector(Constants.SRC,ds_lasingoff.env())
    if xtcav_lasingoff is None:
        print 'No lasing off image found for unixtime',time,'and fiducials',fid
    print 'Found lasing off shot in run',lor.parameters.run
    return xtcav_lasingoff.raw(evt_lasingoff)

import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from xtcav.LasingOnCharacterization import *
XTCAVRetrieval=LasingOnCharacterization(lasingoffreferencepath="/reg/d/psdm/AMO/amox23616/calib/Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/lasingoffreference/60-78.data")
for evt in ds.events():
    if not XTCAVRetrieval.processEvent(evt):
        continue
    gd = gdet.get(evt)
    
    time, power = XTCAVRetrieval.xRayPower(method="COM") 
    agreement = XTCAVRetrieval.reconstructionAgreement()
    print 'Agreement:',agreement,'Gasdet:',gd.f_11_ENRC()
    
    if agreement<0.5: 
        continue

    results=XTCAVRetrieval._pulse_characterization

    xtcav_lasingoff = getLasingOffShot(XTCAVRetrieval,ds.env().experiment())

    raw = xtcav.raw(evt)

    plt.subplot(3,2,1)
    plt.title('Lasing On')
    plt.imshow(raw)

    plt.subplot(3,2,2)
    plt.title('Lasing Off')
    plt.imshow(xtcav_lasingoff)

    plt.subplot(3,2,3)
    plt.title('Current')
    plt.plot(time[0],results.lasingECurrent[0],label='lasing')
    plt.plot(time[0],results.nolasingECurrent[0],label='nolasing')
    #plt.legend()

    plt.subplot(3,2,4)
    plt.title('E (Delta)')
    plt.plot(time[0],results.lasingECOM[0],label='lasing')
    plt.plot(time[0],results.nolasingECOM[0],label='nolasing')
    #plt.legend()

    plt.subplot(3,2,5)
    plt.title('E (Sigma)')
    plt.plot(time[0],results.lasingERMS[0],label='lasing')
    plt.plot(time[0],results.nolasingERMS[0],label='nolasing')
    #plt.legend()

    plt.subplot(3,2,6)
    plt.title('Power')
    plt.plot(time[0],power[0])

    plt.show()


# available quantities from step3, from xtcav/src/Utils.py:ProcessLasingSingleShot

# 't':t,                                  #Master time vector in fs
# 'powerECOM':powerECOM,              #Retrieved power in GW based on ECOM
# 'powerERMS':powerERMS,              #Retrieved power in GW based on ERMS
# 'powerAgreement':powerAgreement,        #Agreement between the two intensities
# 'bunchdelay':bunchdelay,                #Delay from each bunch with respect to the first one in fs
# 'bunchdelaychange':bunchdelaychange,    #Difference between the delay from each bunch with respect to the first one in fs and the same form the non lasing reference
# 'xrayenergy':shotToShot['xrayenergy'],  #Total x-ray energy from the gas detector in J
# 'lasingenergyperbunchECOM': eBunchCOM,  #Energy of the XRays generated from each bunch for the center of mass approach in J
# 'lasingenergyperbunchERMS': eBunchRMS,  #Energy of the XRays generated from each bunch for the dispersion approach in J
# 'bunchenergydiff':bunchenergydiff,                  #Distance in energy for each bunch with respect to the first one in MeV
# 'bunchenergydiffchange':bunchenergydiffchange,      #Comparison of that distance with respect to the no lasing
# 'lasingECurrent':lasingECurrent,        #Electron current for the lasing trace (In #electrons/s)
# 'nolasingECurrent':nolasingECurrent,    #Electron current for the no lasing trace (In #electrons/s)
# 'lasingECOM':lasingECOM,                #Lasing energy center of masses for each time in MeV
# 'nolasingECOM':nolasingECOM,            #No lasing energy center of masses for each time in MeV
# 'lasingERMS':lasingERMS,                #Lasing energy dispersion for each time in MeV
# 'nolasingERMS':nolasingERMS,            #No lasing energy dispersion for each time in MeV
# 'NB': NB,                               #Number of bunches
# 'groupnum': groupnum                    #group number of lasing-off shot
