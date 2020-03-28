#(c) Coded by Alvaro Sanchez-Gonzalez 2014

#Script for the retrieval of the pulses shot to shot

import logging
logger = logging.getLogger(__name__)

import os
import time
import psana
import numpy as np
#import glob
#import pdb
#import IPython
import sys
#import getopt
import math
import warnings
import psana.xtcav.Utils as xtu
import psana.xtcav.UtilsPsana as xtup
import psana.xtcav.SplittingUtils as su
import psana.xtcav.Constants as cons
from   psana.xtcav.DarkBackgroundReference import *
from   psana.xtcav.LasingOffReference import *
from   psana.xtcav.CalibrationPaths import *


class LasingOnCharacterization():

    """
    Class that can be used to reconstruct the full X-Ray power time profile for single or multiple bunches, relying on the presence of a dark background reference, and a lasing off reference. (See GenerateDarkBackground and Generate LasingOffReference for more information)
    Attributes:
        calibration_path (str): Custom calibration directory in case the default is not intended to be used.
        start_image (int): image in run to start from
        snr_filter (float): Number of sigmas for the noise threshold (If not set, the value that was used for the lasing off reference will be used).
        roi_expand (float): number of waists that the region of interest around will span around the center of the trace (If not set, the value that was used for the lasing off reference will be used).
        roi_fraction (float): fraction of pixels that must be non-zero in roi(s) of image for analysis
        island_split_method (str): island splitting algorithm. Set to 'scipylabel' or 'contourLabel'  The defaults parameter is then one used for the lasing off reference or 'scipylabel'.
    """

    def __init__(self, 
        #all parameters defaulted to None since code handles filling parameters later
        num_bunches = None, 
        start_image = 0,
        snr_filter=None,
        roi_expand=None,
        roi_fraction=None,
        island_split_method=None,
        island_split_par1=None,
        island_split_par2=None,
        dark_reference_path=None,
        lasingoff_reference_path=None,
        calibration_path=''
        ):


        #fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s' # '%(message)s'
        fmt='[%(levelname).1s] L%(lineno)04d : %(message)s'
        logging.basicConfig(format=fmt, datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)


        #Handle warnings
        warnings.filterwarnings('always',module='Utils',category=UserWarning)
        warnings.filterwarnings('ignore',module='Utils',category=RuntimeWarning, message="invalid value encountered in divide")
        
        self.num_bunches = num_bunches              #Number of bunches
        self.start_image = start_image
        self.snr_filter = snr_filter                  #Number of sigmas for the noise threshold
        self.roi_expand = roi_expand                  #Parameter for the roi location
        self.roi_fraction = roi_fraction               
        self.island_split_method = island_split_method  #Method for island splitting
        self.island_split_par1 = island_split_par1
        self.island_split_par2 = island_split_par2
        
        self.dark_reference_path = dark_reference_path  #Dark reference file path
        self.lasingoff_reference_path = lasingoff_reference_path        #Lasing off reference file path 
        self.calibration_path = calibration_path
        
        self._envset = False
        self._calibrationsset = False

        self._setDataSource

        self._loadDarkReference()
        self._loadLasingOffReference()

            
    def _setDataSource(self):
        """
        Method that uses detector interface to gather data source info. This method is called automatically and should not be called by the user unless he has a knowledge of the operation done by this class internally.    
        """
        try:
            self._env = psana.det_interface._getEnv()
        except RuntimeError:
            #warnings.warn_explicit('Data source not set yet. Initialize data source before starting analysis',UserWarning,'XTCAV',0)
            return

        self._xtcav_camera = psana.Detector(cons.DETNAME)
        self._ebeam_data = psana.Detector(cons.EBEAM)
        self._gasdetector_data = psana.Detector(cons.GAS_DETECTOR)
        self._ebeam = None
        self._gasdetector = None
        
        self._envset = True


    def _setCalibrations(self, evt):
        """
        Method that sets the xtcav calibration values for a given run. This method is called automatically and should not be called by the user unless he has a knowledge of the operation done by this class internally.    
        """
        self._currentrun = evt.run()
        if not self._darkreference:
            self._loadDarkReference()
        if not self._lasingoffreference:
            self._loadLasingOffReference()

        self._roixtcav = xtup.getXTCAVImageROI(evt)
        self._global_calibration = xtup.getGlobalXTCAVCalibration(evt)
        self._saturation_value = xtup.getCameraSaturationValue(evt)
        if self._roixtcav and self._global_calibration and self._saturation_value:
            self._calibrationsset = True

        #Only reason to do this is to allow us to use same 'processImage' function across lasing on/off shots
        self.parameters = LasingOnParameters(self.num_bunches, self.snr_filter,  self.roi_expand,
            self.roi_fraction, self.island_split_method, self.island_split_par1, self.island_split_par2 )


    def _loadDarkReference(self):
        """
        Method that loads the dark reference. This method is called automatically and should not be called by the user unless he has a knowledge of the operation done by this class internally.    
        """
        self._darkreference = None

        if not self.dark_reference_path:
            if not self._envset:
                return 

            cp = CalibrationPaths(self._env, self.calibration_path)       
            self.dark_reference_path = cp.findCalFileName(cons.DB_FILE_NAME, self._currentrun)
            #If we could not find it, we just wont use it, and return False
            if not self.dark_reference_path:
                warnings.warn_explicit('Dark reference for run %d not found, image will not be background substracted' % self._currentevent.run(),UserWarning,'XTCAV',0)
                return    
            print("Using file " + self.dark_reference_path.split("/")[-1] + " for dark reference")
        
        self._darkreference = DarkBackgroundReference.load(self.dark_reference_path)

                
    def _loadLasingOffReference(self):
        """
        Method that loads the lasing off reference. This method is called automatically and should not be called by the user unless he has a knowledge of the operation done by this class internally.
        """
        self._lasingoffreference = None

        if not self.lasingoff_reference_path:
            if not self._envset:
                return 

            cp = CalibrationPaths(self._env, self.calibration_path)     
            self.lasingoff_reference_path = cp.findCalFileName(cons.LOR_FILE_NAME,  self._currentrun)
            
            if self.lasingoff_reference_path:
                self._lasingoffreference = LasingOffReference.load(self.lasingoff_reference_path)

        if not self._lasingoffreference:
            warnings.warn_explicit('Lasing off reference for run %d not found, using set or default values for image processing' % self._currentevent.run(),UserWarning,'XTCAV',0)
            self._loadDefaultProcessingParameters()
        else:
            print("Using file " + self.lasingoff_reference_path.split("/")[-1] + " for lasing off reference")
            self._loadLasingOffReferenceParameters()

            
    def _loadDefaultProcessingParameters(self):
        """
        Method that sets some standard processing parameters in case they have not been explicitly set by the user and could not been retrieved from the lasing off reference. This method is called automatically and should not be called by the user unless he has a knowledge of the operation done by this class internally.             
        """
        if not self.num_bunches:
            self.num_bunches=1
        if not self.snr_filter:
            self.snr_filter=10
        if not self.roi_expand:
            self.roi_expand=2.5 
        if not self.roi_fraction:
            self.roi_fraction=cons.ROI_PIXEL_FRACTION    
        if not self.island_split_method:
            self.island_split_method=cons.DEFAULT_SPLIT_METHOD       
        if not self.island_split_par1:        
            self.island_split_par1=3.0
        if not self.island_split_par2:        
            self.island_split_par2=5.0


    def _loadLasingOffReferenceParameters(self):
        """
        Method that sets processing parameters from the lasing off reference in case they have not been explicitly set by the user (except for the number of bunches. That one is must match). This method is called automatically and should not be called by the user unless he has a knowledge of the operation done by this class internally.             
        """
        if self.num_bunches and self.num_bunches != self._lasingoffreference.parameters.num_bunches:
            warnings.warn_explicit('Number of bunches input (%d) differs from number of bunches found in lasing off reference (%d). Overwriting input value.' % (self.num_bunches,self._lasingoffreference.parameters.num_bunches) ,UserWarning,'XTCAV',0)
        self.num_bunches=self._lasingoffreference.parameters.num_bunches
        if not self.snr_filter:
            self.snr_filter=self._lasingoffreference.parameters.snr_filter
        if not self.roi_expand:
            self.roi_expand=self._lasingoffreference.parameters.roi_expand
        if not self.roi_fraction:
            self.roi_fraction=self._lasingoffreference.parameters.roi_fraction
        if not self.dark_reference_path:
            self.dark_reference_path=self._lasingoffreference.parameters.dark_reference_path
        if not self.island_split_method:
            self.island_split_method=self._lasingoffreference.parameters.island_split_method
        if not self.island_split_par1:        
            self.island_split_par1=self._lasingoffreference.parameters.island_split_par1
        if not self.island_split_par2:        
            self.island_split_par2=self._lasingoffreference.parameters.island_split_par2 


    def processEvent(self, evt):
        """
        Args:
            evt (psana event): relevant event to retrieve information from
            
        Returns:
            True: All the input form detectors necessary for a good reconstruction are present in the event. 
            False: The information from some detectors is missing for that event. It may still be possible to get information.
        """
        self._currentevent = evt
         #Reset image results
        self._pulse_characterization = None
        self._image_profile = None
        self._processed_image = None

        if not self._envset:
            self._setDataSource()

        if not self._envset:
            warnings.warn_explicit('Data source not set yet. Initialize data source before starting analysis',UserWarning,'XTCAV',0)
            return False


        if not self._calibrationsset:
            self._setCalibrations(evt)
            if not self._calibrationsset:
                return False

        self._ebeam = self._ebeam_data.get(evt)
        self._gasdetector = self._gasdetector_data.get(evt)

        shot_to_shot = xtup.getShotToShotParameters(self._ebeam, self._gasdetector, evt.get(psana.EventId)) #Obtain the shot to shot parameters necessary for the retrieval of the x and y axis in time and energy units
        
        if not shot_to_shot.valid: #If the information is not good, we skip the event
            return False 
       
        self._rawimage = self._xtcav_camera.image(evt)

        if self._rawimage is None: 
            warnings.warn_explicit('Could not retrieve image',UserWarning,'XTCAV',0)
            return False

        self._image_profile, self._processed_image =  xtu.processImage(self._rawimage, self.parameters, self._darkreference, self._global_calibration, 
                                                    self._saturation_value, self._roixtcav, shot_to_shot)
        if not self._image_profile:
            warnings.warn_explicit('Cannot create image profile',UserWarning,'XTCAV',0)
            return False

        if not self._lasingoffreference:
            warnings.warn_explicit('Cannot perform analysis without lasing off reference',UserWarning,'XTCAV',0)
            return False

        #Using all the available data, perform the retrieval for that given shot        
        self._pulse_characterization = xtu.processLasingSingleShot(self._image_profile, self._lasingoffreference.averaged_profiles) 
        return True if self._pulse_characterization else False

        
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
            warnings.warn_explicit('Image profile not created for current event due to issues with image',UserWarning,'XTCAV',0)
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
            warnings.warn_explicit('Pulse characterization not created for current event due to issues with image',UserWarning,'XTCAV',0)
            
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
            warnings.warn_explicit('Pulse characterization not created for current event due to issues with image. ' +\
                'Cannot construct pulse delay',UserWarning,'XTCAV',0)
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
                warnings.warn_explicit('Method %s not supported' % (method),UserWarning,'XTCAV',0)
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
            warnings.warn_explicit('Pulse characterization not created for current event due to issues with image. ' +\
                'Cannot construct pulse FWHM',UserWarning,'XTCAV',0)
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
                warnings.warn_explicit('Method %s not supported' % (method),UserWarning,'XTCAV',0)
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
            warnings.warn_explicit('Image profile not created for current event due to issues with image. ' +\
                'Cannot construct inter bunch pulse delay',UserWarning,'XTCAV',0)
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
            warnings.warn_explicit('Image profile not created for current event due to issues with image. ' +\
                'Cannot construct inter bunch pulse delay',UserWarning,'XTCAV',0)
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
            warnings.warn_explicit('Image profile not created for current event due to issues with image. ' +\
                'Cannot construct inter bunch pulse delay',UserWarning,'XTCAV',0)
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
            warnings.warn_explicit('Image profile not created for current event due to issues with image. ' +\
                'Cannot construct electron current',UserWarning,'XTCAV',0)
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
            warnings.warn_explicit('Pulse characterization not created for current event due to issues with image. ' +\
                'Cannot construct pulse FWHM',UserWarning,'XTCAV',0)
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
            warnings.warn_explicit('Method %s not supported' % (method),UserWarning,'XTCAV',0)
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
            warnings.warn_explicit('Pulse characterization not created for current event due to issues with image. ' +\
                'Cannot construct pulse FWHM',UserWarning,'XTCAV',0)
            return None
        
        if method=='RMS':
            energyperbunch = self._pulse_characterization.lasingenergyperbunchERMS
        elif method=='COM':
            energyperbunch = self._pulse_characterization.lasingenergyperbunchECOM
        else:
            warnings.warn_explicit('Method %s not supported' % (method),UserWarning,'XTCAV',0)
            return None
       
        return energyperbunch  
        
    
    def processedXTCAVImage(self):    
        """
        Method which returns the processed XTCAV image after background subtraction, noise removal, region of interest cropping and multiple bunch separation. This does not require a lasing off reference.

        Returns: 
            3D array where the first index is bunch number, and the other two are the image.
        """     
        if self._processed_image is None:
            warnings.warn_explicit('Image not processed for current event due to issues with image. ' +\
                'Returning raw image',UserWarning,'XTCAV',0)
            return self._rawimage
          
        return self._processed_image


    def rawXTCAVImage(self):
        """
        Method which returns the processed XTCAV image after background subtraction, noise removal, region of interest cropping and multiple bunch separation. This does not require a lasing off reference.

        Returns: 
            3D array where the first index is bunch number, and the other two are the image.
        """     
        if self._rawimage is None:
            warnings.warn_explicit('Image not processed for current event due to issues with image. ' +\
                'Returning raw image',UserWarning,'XTCAV',0)
        return self._rawimage
          

    def processedXTCAVImageROI(self):    
        """
        Method which returns the position of the processed XTCAV image within the whole CCD after background subtraction, noise removal, region of interest cropping and multiple bunch separation. This does not require a lasing off reference.

        Returns: 
            Dictionary with the region of interest parameters.
        """     
        if self._processed_image is None:
            warnings.warn_explicit('Image profile not created for current event due to issues with image.',UserWarning,'XTCAV',0)
            return None
            
        return self._image_profile.roi


    def processedXTCAVImageProfile(self):    
        """
        Method which returns the position of the processed XTCAV image within the whole CCD after background subtraction, noise removal, region of interest cropping and multiple bunch separation. This does not require a lasing off reference.

        Returns: 
            Dictionary with the region of interest parameters.
        """     
        if self._image_profile is None:
            warnings.warn_explicit('Image profile not created for current event due to issues with image.',UserWarning,'XTCAV',0)
            return None
            
        return self._image_profile

        
    def reconstructionAgreement(self): 
        """
        Value for the agreement of the reconstruction using the RMS method and using the COM method. It consists of a value ranging from -1 to 1.

        Returns: 
            value for the agreement.
        """
        if not self._pulse_characterization:
            warnings.warn_explicit('Pulse characterization not created for current event due to issues with image. ' +\
                'Cannot calculate reconstruction agreement',UserWarning,'XTCAV',0)
            return 0
                       
        return np.mean(self._pulse_characterization.powerAgreement)  

LasingOnParameters = xtu.namedtuple('LasingOnParameters', 
    ['num_bunches', 
    'snr_filter', 
    'roi_expand',
    'roi_fraction', 
    'island_split_method',
    'island_split_par1', 
    'island_split_par2'])   
        
