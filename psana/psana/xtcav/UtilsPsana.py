
import logging
logger = logging.getLogger(__name__)

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

def get_attribute(o, aname='valsxtp'):
    """ wrapper for getattr bwith message to the logger.
    """
    oat = getattr(o, aname, None)
    if oat is None : 
        logger.warning('Object %s does not contain attribute %s' % (str(o), attrname))
        return None
    logger.debug('XXX dir(oat): %s' % str(dir(oat)))
    return oat


def getCameraSaturationValue(valsxtp, evt):
    try:
        analysis_version = get_attribute(valsxtp, 'valsxtp') #SimulatorDetector(cons.ANALYSIS_VERSION)
        if analysis_version(evt) is not None:
            return (1<<12)-1
    except:
        pass

    return (1<<14)-1
    

def getGlobalXTCAVCalibration(ovals, evt):
    """
    Obtain the global XTCAV calibration form the epicsStore
    Arguments:
      xtcavpars, evt
    Output:
      globalCalibration: struct with the parameters
      ok: if all the data was retrieved correctly
    """
    def getCalibrationValue(ovals, parnames, evt):
        val = None
        for pname in parnames:
            ov = getattr(ovals, pname, None)
            #print('parameter name: %s object value %s' % (pname, str(ov)))
            if ov is None : continue
            val = ov(evt)
            if abs(val) < 1e-100 : continue
            return val
        if val is None :
            logger.warning('XTCAV variable/s %s is/are not found in the xtc event' % str(parnames))
        return None

    global_calibration = GlobalCalibration(
        umperpix    =getCalibrationValue(ovals, cons.UM_PER_PIX_names, evt), 
        strstrength =getCalibrationValue(ovals, cons.STR_STRENGTH_names, evt), 
        rfampcalib  =getCalibrationValue(ovals, cons.RF_AMP_CALIB_names, evt), 
        rfphasecalib=getCalibrationValue(ovals, cons.RF_PHASE_CALIB_names, evt), 
        dumpe       =getCalibrationValue(ovals, cons.DUMP_E_names, evt), 
        dumpdisp    =getCalibrationValue(ovals, cons.DUMP_DISP_names, evt)
    )

    #self check
    for k,v in global_calibration._asdict().items():
        if not v:
            logger.warning('No XTCAV Calibration for epics variable ', k)
            return None

    return global_calibration
                          

def getXTCAVImageROI(ovals, evt):
    """TBD: need to implement loop over list of variable names, like in original version.
    """

    for i in range(len(cons.ROI_SIZE_X_names)):
        try:
            #roiXN=SimulatorDetector(cons.ROI_SIZE_X_names[i])
            #roiX =SimulatorDetector(cons.ROI_START_X_names[i])
            #roiYN=SimulatorDetector(cons.ROI_SIZE_Y_names[i])
            #roiY =SimulatorDetector(cons.ROI_START_Y_names[i])

            xN = ovals.XTCAV_ROI_sizeX(evt)   # roiXN(evt)  #Size of the image in X                           
            x0 = ovals.XTCAV_ROI_startX(evt)  # roiX(evt)   #Position of the first pixel in x
            yN = ovals.XTCAV_ROI_sizeY(evt)   # roiYN(evt)  #Size of the image in Y 
            y0 = ovals.XTCAV_ROI_startY(evt)  # roiY(evt)   #Position of the first pixel in y

            #xN = cons.ROI_SIZE_X  #Size of the image in X                           
            #x0 = cons.ROI_START_X #Position of the first pixel in x
            #yN = cons.ROI_SIZE_Y  #Size of the image in Y 
            #y0 = cons.ROI_START_Y #Position of the first pixel in y

            x = x0+np.arange(0, xN) 
            y = y0+np.arange(0, yN) 

            return ROIMetrics(xN, x0, yN, y0, x, y) 

        except KeyError:
            continue
        
    warnings.warn_explicit('No XTCAV ROI info 2',UserWarning,'XTCAV',0)
    return None


def getShotToShotParameters(ebeam, gasdetector, evt_id):
    time = evt_id.time()
    sec  = time[0]
    nsec = time[1]
    unixtime = int((sec<<32)|nsec)
    fiducial = evt_id.fiducials()

    energydetector = cons.ENERGY_DETECTOR
 
    if ebeam:    
        ebeamcharge = ebeam.ebeamCharge()
        xtcavrfamp  = ebeam.ebeamXTCAVAmpl()
        xtcavrfphase= ebeam.ebeamXTCAVPhase()
        dumpecharge = ebeam.ebeamDumpCharge()*cons.E_CHARGE #In C 
        
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
        


#def divideImageTasks(first_image, last_image, rank, size):
#    """
#    DEPRICATED
#    Split image numbers among cores based on number of cores and core ID
#    The run will be segmented into chunks of 4 shots, with each core alternatingly assigned to each.
#    e.g. Core 1 | Core 2 | Core 3 | Core 1 | Core 2 | Core 3 | ....
#    """
#    num_shots = last_image - first_image
#    if num_shots <= 0:
#        return np.empty()
#    tiling = np.arange(rank*4, rank*4+4,1) #  returns [0, 1, 2, 3] if e.g. rank == 0 and size == 4:
#    comb1 = np.tile(tiling, np.ceil(num_shots/(4.*size)).astype(int))  # returns [0, 1, 2, 3, 0, 1, 2, 3, ...]        
#    comb2 = np.repeat(np.arange(0, np.ceil(num_shots/(4.*size)), 1), 4) # returns [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, ...]
#            #  list of shot numbers assigned to this core
#    main = comb2*4*size + comb1  + first_image # returns [  0.   1.   2.   3.  16.  17.  18.  19.  32.  33. ... ]
#    main = np.delete(main, np.where(main>=last_image) )  # remove element if greater or equal to maximum number of shots in run
#    return main.astype(int)

