import numpy as np
import psana
import warnings
import time
from psana.xtcav.Utils import ROIMetrics, GlobalCalibration, ShotToShotParameters
import psana.xtcav.Constants as cons

from psana.xtcav.Simulators import\
  SimulatorDetector
#  SimulatorEBeam,\
#  SimulatorGasDetector,\
#  SimulatorEventId,\
#  SimulatorEnvironment



def getCameraSaturationValue(evt):
    try:
        analysis_version = SimulatorDetector(cons.ANALYSIS_VERSION)
        if analysis_version(evt) is not None:
            return (1<<12)-1
    except:
        pass

    return (1<<14)-1
    

def getGlobalXTCAVCalibration(run, evt):
    """
    Obtain the global XTCAV calibration form the epicsStore
    Arguments:
      epicsStore
    Output:
      globalCalibration: struct with the parameters
      ok: if all the data was retrieved correctly
    """
    def getCalibrationValues(possible_detector_names):
        for i in range(len(possible_detector_names)):
            try:
                det = SimulatorDetector(possible_detector_names[i])
                val = det()
                if abs(val) < 1e-100:
                    continue
                return val 
            except KeyError:
                continue
        return None

    global_calibration = GlobalCalibration(
        umperpix    =getCalibrationValues(cons.UM_PER_PIX_names), 
        strstrength =getCalibrationValues(cons.STR_STRENGTH_names), 
        rfampcalib  =getCalibrationValues(cons.RF_AMP_CALIB_names), 
        rfphasecalib=getCalibrationValues(cons.RF_PHASE_CALIB_names), 
        dumpe       =getCalibrationValues(cons.DUMP_E_names), 
        dumpdisp    =getCalibrationValues(cons.DUMP_DISP_names)
    )
        
    for k,v in global_calibration._asdict().items():
        if not v:
            warnings.warn_explicit('No XTCAV Calibration for epics variable ' + k, UserWarning,'XTCAV',0)
            return None

    return global_calibration
                          

def getXTCAVImageROI(run, evt):

    for i in range(len(cons.ROI_SIZE_X_names)):
        try:
            roiXN=SimulatorDetector(cons.ROI_SIZE_X_names[i])
            roiX =SimulatorDetector(cons.ROI_START_X_names[i])
            roiYN=SimulatorDetector(cons.ROI_SIZE_Y_names[i])
            roiY =SimulatorDetector(cons.ROI_START_Y_names[i])

            xN = roiXN(evt)  #Size of the image in X                           
            x0 = roiX(evt)   #Position of the first pixel in x
            yN = roiYN(evt)  #Size of the image in Y 
            y0 = roiY(evt)   #Position of the first pixel in y

            #xN = cons.ROI_SIZE_X  #Size of the image in X                           
            #x0 = cons.ROI_START_X #Position of the first pixel in x
            #yN = cons.ROI_SIZE_Y  #Size of the image in Y 
            #y0 = cons.ROI_START_Y #Position of the first pixel in y

            x = x0+np.arange(0, xN) 
            y = y0+np.arange(0, yN) 

            return ROIMetrics(xN, x0, yN, y0, x, y) 

        except KeyError:
            continue
        
    warnings.warn_explicit('No XTCAV ROI info',UserWarning,'XTCAV',0)
    return None


def getShotToShotParameters(ebeam, gasdetector, evt_id):
    time = evt_id.time()
    sec  = time[0]
    nsec = time[1]
    unixtime = int((sec<<32)|nsec)
    fiducial = evt_id.fiducials()

    energydetector = cons.ENERGY_DETECTOR
 
    if ebeam:    
        ebeamcharge=ebeam.ebeamCharge()
        xtcavrfamp=ebeam.ebeamXTCAVAmpl()
        xtcavrfphase=ebeam.ebeamXTCAVPhase()
        dumpecharge=ebeam.ebeamDumpCharge()*cons.E_CHARGE #In C 
        
        if gasdetector:
            energydetector=(gasdetector.f_11_ENRC()+gasdetector.f_12_ENRC())/2 
            return ShotToShotParameters(ebeamcharge = ebeamcharge, 
                xtcavrfphase = xtcavrfphase, xtcavrfamp = xtcavrfamp, 
                dumpecharge = dumpecharge, xrayenergy = 1e-3*energydetector, 
                unixtime = unixtime, fiducial = fiducial)     
        else:   
            warnings.warn_explicit('No gas detector info',UserWarning,'XTCAV',0)
                
    else:    
        warnings.warn_explicit('No ebeamv info',UserWarning,'XTCAV',0)
    
    return ShotToShotParameters(unixtime = unixtime, fiducial = fiducial, valid = 0)
        


def divideImageTasks(first_image, last_image, rank, size):
    """
    Split image numbers among cores based on number of cores and core ID
    The run will be segmented into chunks of 4 shots, with each core alternatingly assigned to each.
    e.g. Core 1 | Core 2 | Core 3 | Core 1 | Core 2 | Core 3 | ....
    """
    num_shots = last_image - first_image
    if num_shots <= 0:
        return np.empty()
    tiling = np.arange(rank*4, rank*4+4,1) #  returns [0, 1, 2, 3] if e.g. rank == 0 and size == 4:
    comb1 = np.tile(tiling, np.ceil(num_shots/(4.*size)).astype(int))  # returns [0, 1, 2, 3, 0, 1, 2, 3, ...]        
    comb2 = np.repeat(np.arange(0, np.ceil(num_shots/(4.*size)), 1), 4) # returns [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, ...]
            #  list of shot numbers assigned to this core
    main = comb2*4*size + comb1  + first_image # returns [  0.   1.   2.   3.  16.  17.  18.  19.  32.  33. ... ]
    main = np.delete(main, np.where(main>=last_image) )  # remove element if greater or equal to maximum number of shots in run
    return main.astype(int)

