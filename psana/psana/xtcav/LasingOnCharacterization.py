
""" (c) Coded by Alvaro Sanchez-Gonzalez 2014
    2020-04-06 adopted to LCLS2 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

import os
import sys
import time
import numpy as np
import math

from psana import DataSource

import psana.xtcav.Utils as xtu
import psana.xtcav.UtilsPsana as xtup
import psana.xtcav.SplittingUtils as su
import psana.xtcav.Constants as cons
from   psana.xtcav.DarkBackgroundReference import DarkBackgroundReference
from   psana.xtcav.LasingOffReference import LasingOffReference
from   psana.xtcav.CalibrationPaths import CalibrationPaths
#from psana.pscalib.calib.XtcavUtils import dict_from_xtcav_calib_object, xtcav_calib_object_from_dict
from psana.pyalgos.generic.NDArrUtils import info_ndarr, print_ndarr

class LasingOnCharacterization():
    """
    Class reconstructs the full X-Ray power time profile for single or multiple bunches, relying on the presence of a dark background reference, and a lasing off reference. (See DarkBackgroundReference and LasingOffReference for more information)
    Attributes:
        calibration_path (str): Custom calibration directory in case the default is not intended to be used.
        start_image (int): image in run to start from
        snr_filter (float): Number of sigmas for the noise threshold (If not set, the value that was used for the lasing off reference will be used).
        roi_expand (float): number of waists that the region of interest around will span around the center of the trace (If not set, the value that was used for the lasing off reference will be used).
        roi_fraction (float): fraction of pixels that must be non-zero in roi(s) of image for analysis
        island_split_method (str): island splitting algorithm. Set to 'scipylabel' or 'contourLabel'  The defaults parameter is then one used for the lasing off reference or 'scipylabel'.
    """

    def __init__(self, args, run, dets):
        """
           Arguments:
           - args (argparse.Namespace): container of input parameters as attributes
           - run  (psana.psexp.run.RunSingleFile): object for single run
        """

        self.args = args
        self.run  = run
        self.dets = dets

        #all parameters defaulted to None since code handles filling parameters later
        self.num_bunches         = getattr(args, 'num_bunches', None)
        self.start_image         = getattr(args, 'start_image', 0)
        self.snr_filter          = getattr(args, 'snr_filter', None)          #Number of sigmas for the noise threshold
        self.roi_expand          = getattr(args, 'roi_expand', None)          #Parameter for the roi location
        self.roi_fraction        = getattr(args, 'roi_fraction', None)
        self.island_split_method = getattr(args, 'island_split_method', None) #Method for island splitting
        self.island_split_par1   = getattr(args, 'island_split_par1', None)
        self.island_split_par2   = getattr(args, 'island_split_par2', None)
        self.dark_reference_path = getattr(args, 'dark_reference_path', None) #Dark reference file path
        self.lasingoff_ref_path  = getattr(args, 'lasingoff_reference_path', None) #Lasing off reference file path
        #self.calibration_path   = getattr(args, 'calibration_path', '')

        self._setDetectorDataObjects()
        self._loadDarkReference()
        self._loadLasingOffReference()

        self._calibrationsset = False


    def _setDetectorDataObjects(self):
        """ initialization of detectrs data objects is moved outside class

        run = self.run
        self._camera      = run.Detector(cons.DETNAME)
        self._ebeam       = run.Detector(cons.EBEAM)
        self._gasdetector = run.Detector(cons.GAS_DETECTOR)
        self._eventid     = run.Detector(cons.EVENTID)
        self._xtcavpars   = run.Detector(cons.XTCAVPARS)

        self._camraw   = xtup.get_attribute(self._camera,      'raw')
        self._valsebm  = xtup.get_attribute(self._ebeam,       'valsebm')
        self._valsgd   = xtup.get_attribute(self._gasdetector, 'valsgd')
        self._valseid  = xtup.get_attribute(self._eventid,     'valseid')
        self._valsxtp  = xtup.get_attribute(self._xtcavpars,   'valsxtp')

        if None in (self._camraw, self._valsebm, self._valsgd, self._valseid, self._valsxtp) : 
            sys.error('FATAL ERROR IN THE DETECTOR INTERFACE: MISSING ATTRIBUTE MUST BE IMPLEMENTED')
        """

        #logger.debug('dir(dets): %s', str(dir(self.dets)))
        attrs = [name for name in dir(self.dets) if name[:2] != '__']
        logger.debug('set detectors and data attributes: %s', str(attrs))
        for name in attrs : setattr(self, name, getattr(self.dets, name, None))


    def _loadDarkReference(self):
        """ Loads the dark reference from file or DB.
        """
        self._darkreference = None
        if self.dark_reference_path :
            self._darkreference = DarkBackgroundReference.load(self.dark_reference_path)
            logger.info('Using file ' + self.dark_reference_path.split('/')[-1] + ' for dark reference')

        if self._darkreference is None :
           #dark_data, dark_meta = self._camera.calibconst.get('xtcav_pedestals')
            dark_data, dark_meta = xtup.get_calibconst(self._camera, 'xtcav_pedestals', cons.DETNAME, self.run.expt, self.run.runnum)

            self._darkreference = xtu.xtcav_calib_object_from_dict(dark_data)
            logger.debug('==== dark_meta:\n%s' % str(dark_meta))
            logger.debug('==== dir(_darkreference):\n%s'% str(dir(self._darkreference)))
            logger.debug('==== _darkreference.ROI:\n%s'% str(self._darkreference.ROI))
            logger.debug(info_ndarr(self._darkreference.image, '==== darkreference.image:'))
            logger.info('Using dark reference from DB')

                
    def _loadLasingOffReference(self):
        """ Loads the lasing off reference parameters from file or DB or set them to default.
        """
        self._lasingoffreference = None
            
        if self.lasingoff_ref_path:
            self._lasingoffreference = LasingOffReference.load(self.lasingoff_ref_path)
            logger.info('Using lasing off reference from file %s'%self.lasingoff_ref_path.split('/')[-1])
            self._setLasingOffReferenceParameters()
            return

        if self._lasingoffreference is None:
            #lofr_data, lofr_meta = self._camera.calibconst.get('xtcav_lasingoff')
            lofr_data, lofr_meta = xtup.get_calibconst(self._camera, 'xtcav_lasingoff', cons.DETNAME, self.run.expt, self.run.runnum)
            self._lasingoffreference = xtu.xtcav_calib_object_from_dict(lofr_data)
            logger.debug('==== lofr_meta:\n%s' % str(lofr_meta))
            logger.debug('==== dir(_lasingoffreference):\n%s'% str(dir(self._lasingoffreference)))
            logger.debug('==== _lasingoffreference.parameters:\n%s'% str(self._lasingoffreference.parameters))
            logger.debug('==== _lasingoffreference.averaged_profiles:\n%s'% str(self._lasingoffreference.averaged_profiles))            
            logger.info('Using lasing off reference from DB')
            self._setLasingOffReferenceParameters()
            return

        if not self._lasingoffreference:
            logger.warning('Lasing off reference for run %d not found, using default values' % self._currentevent.run())
            self._setDefaultProcessingParameters()

            
    def _setDefaultProcessingParameters(self):
        """ Method that sets some standard processing parameters in case they have not been explicitly set by the user 
            and could not been retrieved from the lasing off reference.
        """
        if not self.num_bunches:         self.num_bunches=1
        if not self.snr_filter:          self.snr_filter=10
        if not self.roi_expand:          self.roi_expand=2.5 
        if not self.roi_fraction:        self.roi_fraction=cons.ROI_PIXEL_FRACTION    
        if not self.island_split_method: self.island_split_method=cons.DEFAULT_SPLIT_METHOD       
        if not self.island_split_par1:   self.island_split_par1=3.0
        if not self.island_split_par2:   self.island_split_par2=5.0
        if not self.dark_reference_path: self.dark_reference_path = ''


    def _setLasingOffReferenceParameters(self):
        """ Method that sets processing parameters from the lasing off reference in case they have not been explicitly set by the user
            (except for the number of bunches. That one is must match).
        """
        logger.debug('_lasingoffreference.parameters: %s' % str(self._lasingoffreference.parameters))

        #pars = xtu.xtcav_calib_object_from_dict(self._lasingoffreference.parameters)
        pars = self._lasingoffreference.parameters
        pars_num_bunches = pars.num_bunches

        #self._lasingoffreference.parameters = pars

        if self.num_bunches and self.num_bunches != pars_num_bunches:
            logger.warning('Number of bunches input (%d) differs from number of bunches found in lasing off reference (%d).'\
                           'Overwriting input value.'%(self.num_bunches, pars_num_bunches))
        self.num_bunches=pars_num_bunches
        if not self.snr_filter          : self.snr_filter          = pars.snr_filter
        if not self.roi_expand          : self.roi_expand          = pars.roi_expand
        if not self.roi_fraction        : self.roi_fraction        = pars.roi_fraction
        if not self.island_split_method : self.island_split_method = pars.island_split_method
        if not self.island_split_par1   : self.island_split_par1   = pars.island_split_par1
        if not self.island_split_par2   : self.island_split_par2   = pars.island_split_par2 
        if not self.dark_reference_path : self.dark_reference_path = getattr(pars, 'dark_reference_path', '')


    def _setCalibrations(self, evt):
        """ Method that sets the xtcav calibration values for a given run.
        """
        # DONE in __init__
        #if not self._camera: self._setDetectorDataObjects()
        #if not self._darkreference: self._loadDarkReference()
        #if not self._lasingoffreference: self._loadLasingOffReference()

        self._roixtcav = xtup.getXTCAVImageROI(self._valsxtp, evt)
        logger.debug('_roixtcav: %s' % str(self._roixtcav))

        self._global_calibration = xtup.getGlobalXTCAVCalibration(self._valsxtp, evt)
        logger.debug('_global_calibration: %s' % str(self._global_calibration))

        self._saturation_value = xtup.getCameraSaturationValue(self._valsxtp, evt)
        logger.debug('_saturation_value: %d' % self._saturation_value)

        if self._roixtcav and self._global_calibration and self._saturation_value:
            #Only reason to do this is to allow us to use same 'processImage' function
            #across lasing on/off shots
            self.parameters = LasingOnParameters(\
                self.num_bunches, self.snr_filter,\
                self.roi_expand, self.roi_fraction,\
                self.island_split_method, self.island_split_par1,\
                self.island_split_par2)
            self._calibrationsset = True


    def processEvent(self, evt):
        """
        Args:
            evt (psana event): relevant event to retrieve information from
            
        Returns:
            True: All the input form detectors necessary for a good reconstruction are present in the event. 
            False: The information from some detectors is missing for that event. It may still be possible to get information.
        """

        self._currentevent = evt
        self._pulse_characterization = None
        self._image_profile = None
        self._processed_image = None

        if not self._calibrationsset:
            self._setCalibrations(evt)
            if not self._calibrationsset:
                logger.warning('CALIBRATION IS NOT SET YET..., try next event')
                return False


        #Obtain the shot to shot parameters necessary for the retrieval of the x and y axis in time and energy units
        shot_to_shot = xtup.getShotToShotParameters(evt, self._valsebm, self._valsgd, self._valseid)
        logger.debug('shot_to_shot: %s' % str(shot_to_shot))
        
        if not shot_to_shot.valid:
            logger.warning('shot_to_shot info is not valid')
            return False 

        self._rawimage = self._camraw(evt)
        logger.debug(info_ndarr(self._rawimage, 'camera raw:'))

        if self._rawimage is None: 
            logger.warning('Could not retrieve image')
            return False

        self._image_profile, self._processed_image = xtu.processImage(\
            self._rawimage,\
            self.parameters,\
            self._darkreference,\
            self._global_calibration,\
            self._saturation_value,\
            self._roixtcav,\
            shot_to_shot)

        logger.debug('After xtu.processImage: _image_profile:\n%s' % xtu.info_xtcav_object(self._image_profile))
        logger.debug('After xtu.processImage: _processed_image:\n%s' % info_ndarr(self._processed_image))

        if not self._image_profile:
            logger.warning('Cannot create image profile')
            return False

        if not self._lasingoffreference:
            logger.warning('Cannot perform analysis without lasing off reference')
            return False

        #Using all the available data, perform the retrieval for that given shot        
        self._pulse_characterization = xtu.processLasingSingleShot(self._image_profile, self._lasingoffreference.averaged_profiles) 
        logger.debug('After xtu.processLasingSingleShot: _pulse_characterization:\n%s', xtu.info_xtcav_object(self._pulse_characterization))

        if not self._pulse_characterization : return False

        return True

        
    def physicalUnits(self):
        """
        Method which returns a dictionary based list with the physical units for the cropped image

        Returns: 
            PhysicalUnits: List with the results
                'yMeVPerPix':         Number of MeV per pixel for the vertical axis of the image
                'xfsPerPix':          Number of fs per pixel for the horizontal axis of the image
                'xfs':                Horizontal axis of the image in fs
                'yMeV':               Vertical axis of the image in MeV
        """
    
        if not self._image_profile:
            logger.warning('Image profile not created for current event due to issues with image')
            return None
        
        return self._image_profile.physical_units               
        

    def fullResults(self):
        """
        Method which returns a dictionary based list with the full results of the characterization

        Returns: 
            PulseCharacterization: List with the results
                't':                           Master time vector in fs
                'powerECOM':                    Retrieved power in GW based on ECOM
                'powerERMS':                    Retrieved power in GW based on ERMS
                'powerAgreement':               Agreement between the two intensities
                'bunchdelay':                   Delay from each bunch with respect to the first one in fs
                'bunchdelaychange':             Difference between the delay from each bunch with respect to the first one in fs and the same form the non lasing reference
                'xrayenergy':                   Total x-ray energy from the gas detector in J
                'lasingenergyperbunchECOM':     Energy of the XRays generated from each bunch for the center of mass approach in J
                'lasingenergyperbunchERMS':     Energy of the XRays generated from each bunch for the dispersion approach in J
                'bunchenergydiff':              Distance in energy for each bunch with respect to the first one in MeV
                'bunchenergydiffchange':        Comparison of that distance with respect to the no lasing
                'lasingECurrent':               Electron current for the lasing trace (In #electrons/s)
                'nolasingECurrent':             Electron current for the no lasing trace (In #electrons/s)
                'lasingECOM':                   Lasing energy center of masses for each time in MeV
                'nolasingECOM':                 No lasing energy center of masses for each time in MeV
                'lasingERMS':                   Lasing energy dispersion for each time in MeV
                'nolasingERMS':                 No lasing energy dispersion for each time in MeV
                'num_bunches':                           Number of bunches
        """
        if not self._pulse_characterization:
            logger.warning('Pulse characterization not created for current event due to issues with image')
            
        return self._pulse_characterization 

            
    def pulseDelay(self, method='COM'):    
        """
        Method which returns the time of lasing for each bunch based on the x-ray reconstruction. They delays are referred to the center of mass of the total current. The order of the delays goes from higher to lower energy electron bunches.
        Args:
            method (str): method to use to obtain the power profile. 'RMS' or 'COM' 
        Returns: 
            List of the delays for each bunch.
        """
        if not self._pulse_characterization:
            logger.warning('Pulse characterization not created for current event due to issues with image. ' +\
                'Cannot construct pulse delay')
            return None
            
        num_bunches = self._pulse_characterization.num_bunches
        if num_bunches < 1:
            return np.zeros((num_bunches), dtype=np.float64)
        
                  
        peakpos=np.zeros((num_bunches), dtype=np.float64);
        for j in range(num_bunches):
            t = self._pulse_characterization.t + self._pulse_characterization.bunchdelay[j]
            if method == 'RMS':
                power = self._pulse_characterization.powerERMS[j]
            elif method=='COM':
                power = self._pulse_characterization.powerECOM[j]
            else:
                logger.warning('Method %s not supported' % (method))
                return None      
            #quadratic fit around 5 pixels method
            central=np.argmax(power)
            try:
                fit=np.polyfit(t[central-2:central+3],power[central-2:central+3],2)
                peakpos[j]=-fit[1]/(2*fit[0])
            except:
                print("here")
                return None 
            
        return peakpos

            
    def pulseFWHM(self, method='RMS'):    
        """
        Method which returns the FWHM of the pulse generated by each bunch in fs. It uses the power profile. The order of the widths goes from higher to lower energy electron bunches.
        Args:
            method (str): method to use to obtain the power profile. 'RMS' or 'COM'
        Returns: 
            List of the full widths half maximum for each bunch.
        """
        if not self._pulse_characterization:
            logger.warning('Pulse characterization not created for current event due to issues with image. ' +\
                'Cannot construct pulse FWHM')
            return None
            
        num_bunches = self._pulse_characterization.num_bunches
        if num_bunches < 1:
            return np.zeros((num_bunches), dtype=np.float64)
        
                  
        peakwidth=np.zeros((num_bunches), dtype=np.float64);
        for j in range(num_bunches):
            t = self._pulse_characterization.t + self._pulse_characterization.bunchdelay[j]
            if method == 'RMS':
                power = self._pulse_characterization.powerERMS[j]
            elif method=='COM':
                power = self._pulse_characterization.powerECOM[j]
            else:
                logger.warning('Method %s not supported' % (method))
                return None   
            #quadratic fit around 5 pixels method
            threshold=np.max(power)/2
            abovethrestimes=t[power>=threshold]
            dt=t[1]-t[0]
            peakwidth[j]=abovethrestimes[-1]-abovethrestimes[0]+dt
            
        return peakwidth
      
    def interBunchPulseDelayBasedOnCurrent(self):    
        """
        Method which returns the time of lasing for each bunch based on the peak electron current on each bunch. A lasing off reference is not necessary for this retrieval. The delays are referred to the center of mass of the total current. The order of the delays goes from higher to lower energy electron bunches.

        Returns: 
            List with the delay for each bunch.
        """
        if not self._image_profile:
            logger.warning('Image profile not created for current event due to issues with image. ' +\
                'Cannot construct inter bunch pulse delay')
            return None
            
        # if (self._eventresultsstep1['NB']<1):
        #     return np.zeros((self._eventresultsstep1['NB']), dtype=np.float64)
        
        t = self._image_profile.physical_units.xfs   
          
        peakpos=np.zeros((self.num_bunches), dtype=np.float64);
        for j in range(0,self.num_bunches):
            #highest value method
            #peakpos[j]=t[np.argmax(self._eventresultsstep1['imageStats'][j]['xProfile'])]
            
            #five highest values method
            #ind=np.mean(np.argpartition(-self._eventresultsstep2['imageStats'][j]['xProfile'],5)[0:5]) #Find the position of the 5 highest values
            #peakpos[j]=t[ind]
            
            #quadratic fit around 5 pixels method
            central=np.argmax(self._image_profile.image_stats[j].xProfile)
            try:
                fit=np.polyfit(t[central-2:central+3], self._pulse_characterization.image_stats[j].xProfile[central-2:central+3],2)
                peakpos[j]=-fit[1]/(2*fit[0])
            except:
                return None 
            
        return peakpos

        
    def interBunchPulseDelayBasedOnCurrentMultiple(self, n=1, filterwith=7):    
        """
        Method which returns multiple possible times of lasing for each bunch based on the peak electron current on each bunch. A lasing off reference is not necessary for this retrieval. The delays are referred to the center of mass of the total current. The order of the delays goes from higher to lower energy electron bunches. Then within each bunch the "n" delays are orderer from highest peak current yo lowest peak current.
        Args:
            n (int): number of possible times of lasing (peaks in the electron current) to find per bunch
            filterwith (float): Witdh of the peak that is removed before searching for the next peak in the same bunch
        Returns: 
            List with a list of "n" delays for each bunch.
        """
        if not self._image_profile:
            logger.warning('Image profile not created for current event due to issues with image. ' +\
                'Cannot construct inter bunch pulse delay')
            return None
        
        t = self._image_profile.physical_units.xfs  
          
        peakpos=np.zeros((self.num_bunches,n), dtype=np.float64);
           
        for j in range(0,self.num_bunches):
            profile = self._image_profile.image_stats[j].xProfile.copy()
            for k in range(n):
                #highest value method
                #peakpos[j]=t[np.argmax(self._eventresultsstep1['imageStats'][j]['xProfile'])]
                
                #five highest values method
                #ind=np.mean(np.argpartition(-self._eventresultsstep2['imageStats'][j]['xProfile'],5)[0:5]) #Find the position of the 5 highest values
                #peakpos[j]=t[ind]
                
                #quadratic fit around 5 pixels method
                central = np.argmax(profile)
                try:
                    fit = np.polyfit(t[central-2:central+3],profile[central-2:central+3],2)
                    peakpos[j,k] =- fit[1]/(2*fit[0])
                    filter = 1-np.exp(-(t-peakpos[j,k])**2/(filterwith/(2*np.sqrt(np.log(2))))**2)
                    profile = profile*filter                   
                except:
                    peakpos[j,k] = np.nan
                    if k==0:
                        return None
                
        return peakpos
        
    def interBunchPulseDelayBasedOnCurrentFourierFiltered(self,targetwidthfs=20,thresholdfactor=0):    
        """
        Method which returns the time delay between the x-rays generated from different bunches based on the peak electron current on each bunch. A lasing off reference is not necessary for this retrieval. The delays are referred to the center of mass of the total current. The order of the delays goes from higher to lower energy electron bunches. This method includes a Fourier filter that applies a low pass filter to amplify the feature identified as the lasing part of the bunch, and ignore other peaks that may be higher in amplitude but also higher in width. It is possible to threshold the signal before calculating the Fourier transform to automatically discard peaks that may be sharp, but too low in amplitude to be the right peaks.
        Args:
            targetwidthfs (float): Witdh of the peak to be used for calculating delay
            thresholdfactor (float): Value between 0 and 1 that indicates which threshold factor to apply to filter the signal before calculating the fourier transform
        Returns: 
            List with the delay for each bunch.
        """
        if not self._image_profile:
            logger.warning('Image profile not created for current event due to issues with image. ' +\
                'Cannot construct inter bunch pulse delay')
            return None
             
        t = self._image_profile.physical_units.xfs    
        
        #Preparing the low pass filter
        N = len(t)
        dt = abs(self._image_profile.physical_units.xfsPerPix)
        if dt*N==0:
            return None
        df = 1./(dt*N)
        
        f = np.array(range(0, N/2+1) + range(-N/2+1,0))*df
                           
        ffilter=(1-np.exp(-(f*targetwidthfs)**6))
          
        peakpos=np.zeros((self.num_bunches), dtype=np.float64);
        for j in range(0,self.num_bunches):
            #Getting the profile and the filtered version
            profile = self._image_profile.image_stats[j].xProfile
            profilef = profile-np.max(profile)*thresholdfactor
            profilef[profilef<0] = 0
            profilef = np.fft.ifft(np.fft.fft(profilef)*ffilter)
        
            #highest value method
            #peakpos[j]=t[np.argmax(profilef)]
            
            #five highest values method
            #ind=np.mean(np.argpartition(-profilef,5)[0:5]) #Find the position of the 5 highest values
            #peakpos[j]=t[ind]
            
            #quadratic fit around 5 pixels method and then fit to the original signal
            central=np.argmax(profilef)
            try:
                fit=np.polyfit(t[central-2:central+3],profile[central-2:central+3],2)
                peakpos[j]=-fit[1]/(2*fit[0])
            except:
                return None 
            
        return peakpos

    def quadRefine(self,p):
        x1,x2,x3 = p + np.array([-1,0,1])
        y1,y2,y3 = self.wf[(p-self.rangelim[0]-1):(p-self.rangelim[0]+2)]
        d = (x1-x2)*(x1-x3)*(x2-x3)
        A = ( x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2) ) / d
        B = ( x3**2.0 * (y1-y2) + x2**2.0 * (y3-y1) + x1**2.0 * (y2-y3) ) / d
        return -1*B / (2*A)


    def electronCurrentPerBunch(self):    
        """
        Method which returns the electron current per bunch. A lasing off reference is not necessary for this retrieval.

        Returns: 
            out1: time vectors in fs
            out2: electron currents in arbitrary units
        """
        if not self._image_profile:
            logger.warning('Image profile not created for current event due to issues with image. ' +\
                'Cannot construct electron current')
            return None, None
        
        t = self._image_profile.physical_units.xfs    

        tout = np.zeros((self.num_bunches, len(t)), dtype=np.float64);
        currents = np.zeros((self.num_bunches, len(t)), dtype=np.float64);
        for j in range(0,self.num_bunches):
            tout[j,:]=t
            currents[j,:]=self._image_profile.image_stats[j].xProfile
                    
        return tout, currents
        

    def xRayPower(self, method='RMS'):       
        """
        Method which returns the power profile for the X-Rays generated by each electron bunch. This is the averaged result from the RMS method and the COM method.

        Args:
            method (str): method to use to obtain the power profile. 'RMS' or 'COM' 
        Returns: 
            out1: time vectors in fs. 2D array where the first index refers to bunch number, and the second index to time.
            out2: power profiles in GW. 2D array where the first index refers to bunch number, and the second index to the power profile.
        """

        if not self._pulse_characterization:
            logger.warning('Pulse characterization not created for current event due to issues with image. ' +\
                'Cannot construct pulse FWHM')
            return None, None
                        
        mastert = self._pulse_characterization.t

        t = np.zeros((self.num_bunches, len(mastert)), dtype=np.float64);
        for j in range(self.num_bunches):
            t[j,:] = mastert+self._pulse_characterization.bunchdelay[j]

        if method=='RMS':
            power = self._pulse_characterization.powerERMS
        elif method=='COM':
            power = self._pulse_characterization.powerECOM
        else:
            logger.warning('Method %s not supported' % (method))
            return t, None
            
        return t,power       
        
        
    def xRayEnergyPerBunch(self, method='RMS'):   
        """
        Method which returns the total X-Ray energy generated per bunch. This is the averaged result from the RMS method and the COM method.
        Args:
            method (str): method to use to obtain the power profile. 'RMS' or 'COM' 
        Returns: 
            List with the values of the energy for each bunch in J
        """ 
        if not self._pulse_characterization:
            logger.warning('Pulse characterization not created for current event due to issues with image. ' +\
                'Cannot construct pulse FWHM')
            return None
        
        if method=='RMS':
            energyperbunch = self._pulse_characterization.lasingenergyperbunchERMS
        elif method=='COM':
            energyperbunch = self._pulse_characterization.lasingenergyperbunchECOM
        else:
            logger.warning('Method %s not supported' % (method))
            return None
       
        return energyperbunch  
        
    
    def processedXTCAVImage(self):    
        """
        Method which returns the processed XTCAV image after background subtraction, noise removal, region of interest cropping and multiple bunch separation. This does not require a lasing off reference.

        Returns: 
            3D array where the first index is bunch number, and the other two are the image.
        """     
        if self._processed_image is None:
            logger.warning('Image not processed for current event due to issues with image. ' +\
                'Returning raw image')
            return self._rawimage
          
        return self._processed_image


    def rawXTCAVImage(self):
        """
        Method which returns the processed XTCAV image after background subtraction, noise removal, region of interest cropping and multiple bunch separation. This does not require a lasing off reference.

        Returns: 
            3D array where the first index is bunch number, and the other two are the image.
        """     
        if self._rawimage is None:
            logger.warning('Image not processed for current event due to issues with image. ' +\
                'Returning raw image')
        return self._rawimage
          

    def processedXTCAVImageROI(self):    
        """
        Method which returns the position of the processed XTCAV image within the whole CCD after background subtraction, noise removal, region of interest cropping and multiple bunch separation. This does not require a lasing off reference.

        Returns: 
            Dictionary with the region of interest parameters.
        """     
        if self._processed_image is None:
            logger.warning('Image profile not created for current event due to issues with image.')
            return None
            
        return self._image_profile.roi


    def processedXTCAVImageProfile(self):    
        """
        Method which returns the position of the processed XTCAV image within the whole CCD after background subtraction, noise removal, region of interest cropping and multiple bunch separation. This does not require a lasing off reference.

        Returns: 
            Dictionary with the region of interest parameters.
        """     
        if self._image_profile is None:
            logger.warning('Image profile not created for current event due to issues with image.')
            return None
            
        return self._image_profile

        
    def reconstructionAgreement(self): 
        """
        Value for the agreement of the reconstruction using the RMS method and using the COM method. It consists of a value ranging from -1 to 1.

        Returns: 
            value for the agreement.
        """
        if not self._pulse_characterization:
            logger.warning('Pulse characterization not created for current event due to issues with image. ' +\
                'Cannot calculate reconstruction agreement')
            return 0
                       
        return np.mean(self._pulse_characterization.powerAgreement)  


    def resultsProcessImage(self):
        t, power  = self.xRayPower()  
        agreement = self.reconstructionAgreement()
        pulse     = self.pulseDelay()
        return t, power, agreement, pulse


    def printProcessImageResults(self):
        t, power, agr, pulse = self.resultsProcessImage()
        logger.info('%sAgreement: %.3f%%  Max power: %g  GW Pulse Delay: %.3f '%(12*' ', agr*100,np.amax(power), pulse[0]))


LasingOnParameters = xtu.namedtuple('LasingOnParameters', 
    ['num_bunches', 
    'snr_filter', 
    'roi_expand',
    'roi_fraction', 
    'island_split_method',
    'island_split_par1', 
    'island_split_par2'])   
        
#----------
#----------
#----------
#----------
#----------
#----------

class Empty():
    pass


def setDetectors(run, camera=None, ebeam=None, gasdetector=None, eventid=None, xtcavpars=None):
    """ sets detector and data objects
    """
    o = Empty()
    o._camera      = camera      if camera      is not None else run.Detector(cons.DETNAME)      # 'xtcav'      
    o._ebeam       = ebeam       if ebeam       is not None else run.Detector(cons.EBEAM)        # 'ebeam'      
    o._gasdetector = gasdetector if gasdetector is not None else run.Detector(cons.GAS_DETECTOR) # 'gasdetector'
    o._eventid     = eventid     if eventid     is not None else run.Detector(cons.EVENTID)      # 'eventid'    
    o._xtcavpars   = xtcavpars   if xtcavpars   is not None else run.Detector(cons.XTCAVPARS)    # 'xtcavpars'  

    o._camraw   = xtup.get_attribute(o._camera,      'raw')
    o._valsebm  = xtup.get_attribute(o._ebeam,       'valsebm')
    o._valsgd   = xtup.get_attribute(o._gasdetector, 'valsgd')
    o._valseid  = xtup.get_attribute(o._eventid,     'valseid')
    o._valsxtp  = xtup.get_attribute(o._xtcavpars,   'valsxtp')

    if None in (o._camraw, o._valsebm, o._valsgd, o._valseid, o._valsxtp) : 
        sys.error('FATAL ERROR IN THE DETECTOR INTERFACE: MISSING ATTRIBUTE MUST BE IMPLEMENTED')
    return o


def data_camera(camraw, evt):
    #o = Empty()
    #o.rawimage = camraw(evt)
    return {'rawimage' : camraw(evt)}


def data_ebeam(valsebm, evt):
    return {\
    'ebeamcharge'  : valsebm.Charge(evt),\
    'xtcavrfamp'   : valsebm.XTCAVAmpl(evt),\
    'xtcavrfphase' : valsebm.XTCAVPhase(evt),\
    'dumpecharge'  : valsebm.DumpCharge(evt)*cons.E_CHARGE\
    }


def data_gasdetector(valsgd, evt):
    return {\
    'o.f_11_ENRC' : valsgd.f_11_ENRC(evt),\
    'o.f_12_ENRC' : valsgd.f_12_ENRC(evt),\
    'o.energydetector' : (o.f_11_ENRC + o.f_12_ENRC)/2,\
    }


def data_eventid(valseid, evt):
    return {\
    'time'      : valseid.time(evt),\
    'fiducials' : valseid.fiducials(evt),\
    }


def data_xtcavpars(valsxtp, evt):
    """
    ROI_SIZE_X_names  = ['XTCAV_ROI_sizeX',  'ROI_X_Length', 'OTRS:DMP1:695:SizeX']
    ROI_SIZE_Y_names  = ['XTCAV_ROI_sizeY',  'ROI_Y_Length', 'OTRS:DMP1:695:SizeY']
    ROI_START_X_names = ['XTCAV_ROI_startX', 'ROI_X_Offset', 'OTRS:DMP1:695:MinX']
    ROI_START_Y_names = ['XTCAV_ROI_startY', 'ROI_Y_Offset', 'OTRS:DMP1:695:MinY']
    
    UM_PER_PIX_names     = ['XTCAV_calib_umPerPx','OTRS:DMP1:695:RESOLUTION']
    STR_STRENGTH_names   = ['XTCAV_strength_par_S','Streak_Strength','OTRS:DMP1:695:TCAL_X']
    RF_AMP_CALIB_names   = ['XTCAV_Amp_Des_calib_MV','XTCAV_Cal_Amp','SIOC:SYS0:ML01:AO214']
    RF_PHASE_CALIB_names = ['XTCAV_Phas_Des_calib_deg','XTCAV_Cal_Phase','SIOC:SYS0:ML01:AO215']
    DUMP_E_names         = ['XTCAV_Beam_energy_dump_GeV','Dump_Energy','REFS:DMP1:400:EDES']
    DUMP_DISP_names      = ['XTCAV_calib_disp_posToEnergy','Dump_Disp','SIOC:SYS0:ML01:AO216']

    o.XTCAV_calib_umPerPx  = valsxtp.XTCAV_calib_umPerPx(evt)
    o.XTCAV_strength_par_S = valsxtp.XTCAV_strength_par_S(evt)
    ...
    a = valsxtp.getattr(valsxtp, 'OTRS:DMP1:695:SizeX', None)
    o.OTRS_DMP1_695_SizeX = None if a is None else a(evt) 
    """

    o = Empty()
    for lstname, names in cons.xtcav_varname.items() :
        for name in names :
            a = xtup.get_attribute(valsxtp, name)
            if a is not None :
                value = a(evt)
                o.setattr(name.replace(':','_'), value)
                #convert name like 'UM_PER_PIX_names' to 'umperpix'
                varname = lstname.rstrip('_names').replace('_','').lower()
                if value is not None :
                    o.setattr(varname, value)
    return o


def procEvents(args):

    fname     = getattr(args, 'fname', '/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0137-e000100-xtcav-v2.xtc2')
    max_shots = getattr(args, 'max_shots', 200)
    mode      = getattr(args, 'mode', 'smd')

    ds = DataSource(files=fname)
    run = next(ds.runs())

    dets = setDetectors(run) # NEEDS IN camera, ebeam, gasdetecto, eventid, xtcavpars
    lon = LasingOnCharacterization(args, run, dets)

    nimgs=0
    for nev,evt in enumerate(run.events()):

        img = dets._camraw(evt)
        logger.info('Event %03d' % nev)
        logger.debug(info_ndarr(img, 'camera raw:'))
        if img is None: continue

        if not lon.processEvent(evt): continue

        t, power, agr, pulse = lon.resultsProcessImage()
        print('%sAgreement:%7.3f%%  Max power: %g  GW Pulse Delay: %.3f '%(12*' ', agr*100,np.amax(power), pulse[0]))

        nimgs += 1
        if nimgs>=max_shots: 
            break

#----------

if __name__ == "__main__":
    sys.exit('run it by command: xtcavLasingOn')

#----------
