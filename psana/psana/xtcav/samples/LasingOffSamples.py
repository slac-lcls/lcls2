
import os
import time
import psana
import numpy as np
import glob
import pdb
import IPython
import sys
import getopt
import warnings
import xtcav.Utils as xtu
import xtcav.UtilsPsana as xtup
import xtcav.SplittingUtils as su
import xtcav.Constants
from xtcav.DarkBackground import *
from xtcav.CalibrationPaths import *
from xtcav.FileInterface import Load as constLoad
from xtcav.FileInterface import Save as constSave
import scipy.io as sio

# PP imports
"""
    Cladd that generates a set of lasing off references for XTCAV reconstruction purposes
    Attributes:
        experiment (str): String with the experiment reference to use. E.g. 'amoc8114'
        runs (str): String with a run number, or a run interval. E.g. '123'  '134-156' 145,136'
        maxshots (int): Maximum number of images to use for the references.
        calibrationpath (str): Custom calibration directory in case the default is not intended to be used.
        num_bunches (int): Number of bunches.
        medianfilter (int): Number of neighbours for median filter.
        snrfilter (float): Number of sigmas for the noise threshold.
        num_groups (int): Number of profiles to average together for each reference.
        roiwaistthres (float): ratio with respect to the maximum to decide on the waist of the XTCAV trace.
        roiexpand (float): number of waists that the region of interest around will span around the center of the trace.
        islandsplitmethod (str): island splitting algorithm. Set to 'scipylabel' or 'contourLabel'  The defaults parameter is 'scipylabel'.
"""

class LasingOffSamples(object):

    def __init__(self, 
            experiment='amoc8114', #Experiment label
            maxshots=None,  #Maximum number of valid shots to process
            run_number='86',       #Run number
            darkreferencepath=None, #Dark reference information
            file_name="sample_profiles",
            num_bunches=1,                   #Number of bunches
            num_groups=5 ,           #Number of profiles to average together
            snrfilter=10,           #Number of sigmas for the noise threshold
            roiwaistthres=0.2,      #Parameter for the roi location
            roiexpand=1,          #Parameter for the roi location
            islandsplitmethod = 'scipyLabel',      #Method for island splitting
            islandsplitpar1 = 3.0,  #Ratio between number of pixels between largest and second largest groups when calling scipy.label
            islandsplitpar2 = 5.,   #Ratio between number of pixels between second/third largest groups when calling scipy.label
            calpath=''):

        self.parameters = LasingOffParameters(experiment = experiment,
            maxshots = maxshots, run = run_number, 
            darkreferencepath = darkreferencepath, num_bunches = num_bunches, 
            num_groups=num_groups, roiwaistthres=roiwaistthres, snrfilter=snrfilter,
            roiexpand = roiexpand, islandsplitmethod=islandsplitmethod, islandsplitpar2 = islandsplitpar2,
            islandsplitpar1=islandsplitpar1, calpath=calpath, version=1)


        
        #Handle warnings
        warnings.filterwarnings('always',module='Utils',category=UserWarning)
        warnings.filterwarnings('ignore',module='Utils',category=RuntimeWarning, message="invalid value encountered in divide")

        print 'Lasing off reference'
        print '\t Experiment: %s' % self.parameters.experiment
        print '\t Runs: %s' % self.parameters.run
        print '\t Number of bunches: %d' % self.parameters.num_bunches
        print '\t Dark reference run: %s' % self.parameters.darkreferencepath
        
        #Loading the data, this way of working should be compatible with both xtc and hdf5 files
        self.dataSource = psana.DataSource("exp=%s:run=%s:idx" % (self.parameters.experiment, self.parameters.run))

        #Camera for the xtcav images
        self.xtcav_camera = psana.Detector(xtcav.Constants.SRC)

        #Ebeam type
        self.ebeam_data = psana.Detector(xtcav.Constants.EBEAM)

        #Gas detectors for the pulse energies
        self.gasdetector_data = psana.Detector(xtcav.Constants.GAS_DETECTOR)

        #Stores for environment variables   
        self.epicsStore = self.dataSource.env().epicsStore()
       
        run = self.dataSource.runs().next()
        env = self.dataSource.env()

        self.ROI_XTCAV, self.global_calibration, self.saturation_value, first_image = self._getCalibrationValues(run, self.xtcav_camera)
        self.dark_background = self.getDarkBackground(env)

        num_processed = 0 #Counter for the total number of xtcav images processed within the run        
        times = run.times()
        image_profiles = []

        if not self.parameters.maxshots:
            self.parameters = self.parameters._replace(maxshots=len(times))

        for t in times: 
            evt = run.event(t)

            #ignore shots without xtcav, because we can get incorrect EPICS information (e.g. ROI).  this is
            #a workaround for the fact that xtcav only records epics on shots where it has camera data, as well
            #as an incorrect design in psana where epics information is not stored per-shot (it is in a more global object
            #called "Env")
            ebeam = self.ebeam_data.get(evt)
            gasdetector = self.gasdetector_data.get(evt)

            shot_to_shot = xtup.GetShotToShotParameters(ebeam, gasdetector, evt.get(psana.EventId)) #Obtain the shot to shot parameters necessary for the retrieval of the x and y axis in time and energy units
        
            if not shot_to_shot.valid: #If the information is not good, we skip the event
                continue 

            img = self.xtcav_camera.image(evt)
            image_profile, processed_img = xtu.processImage(img, self.parameters, self.dark_background, self.global_calibration, 
                                                    self.saturation_value, self.ROI_XTCAV, shot_to_shot)

            if not image_profile:
                continue

            data = {}
            data["xprofile"] = image_profile.image_stats[0].xProfile
            data["yprofile"] = image_profile.image_stats[0].yProfile
            data["yCOMslice"] = image_profile.image_stats[0].yCOMslice
            data["yRMSslice"] = image_profile.image_stats[0].yRMSslice
            data["ebeamcharge"] = shot_to_shot.ebeamcharge
            data["dumpe"] = self.global_calibration.dumpe
            data["img"] = processed_img
            data.update(dict(vars(image_profile.physical_units)))

            image_profiles.append(data)
            
            num_processed += 1
            # print core numb and percentage

            if num_processed % 5 == 0:
                sys.stdout.write('%s%.1f %% done, %d / %d' % ('\r', float(num_processed) / self.parameters.maxshots *100, num_processed, self.parameters.maxshots))
                sys.stdout.flush()
            if num_processed >= self.parameters.maxshots:
                sys.stdout.write('\n')
                break
            if num_processed % 1000 == 0:
                sio.savemat(file_name+'_'+str(self.parameters.run)+'_'+str((num_processed-1)/1000)+'.mat', {'data':image_profiles})
                image_profiles = []

        #  here gather all shots in one core, add all lists
        #sio.savemat('lasing_off_'+str(self.parameters.run)+'final.mat', {'data':final})
        np.save(file_name+'_final', image_profiles)


    def getDarkBackground(self, env):
        if not self.parameters.darkreferencepath:
            cp = CalibrationPaths(env, self.parameters.calpath)
            darkreferencepath = cp.findCalFileName('pedestals', int(self.parameters.run))
            if not darkreferencepath:
                print ('Dark reference for run %s not found, image will not be background substracted' % self.parameters.run)
                return None

            self.parameters = self.parameters._replace(darkreferencepath = darkreferencepath)
        print "Using reference path" + self.parameters.darkreferencepath
        return DarkBackground.load(self.parameters.darkreferencepath)

    def Save(self,path):
        # super hacky... allows us to save without overwriting current instance
        instance = copy.deepcopy(self)
        if instance.parameters:
            instance.parameters = dict(vars(instance.parameters))
        constSave(instance,path)

    @staticmethod    
    def Load(path):
        lor = constLoad(path)
        lor.parameters = LasingOffParameters(**lor.parameters)        
        return constLoad(path)

    @staticmethod
    def _getCalibrationValues(run, xtcav_camera):
        roi_xtcav, global_calibration, saturation_value = None, None, None
        times = run.times()

        end_of_images = len(times)
        for t in range(end_of_images):
            evt = run.event(times[t])
            img = xtcav_camera.image(evt)
            # skip if empty image
            if img is None: 
                continue

            roi_xtcav = xtup.GetXTCAVImageROI(evt)
            global_calibration = xtup.GetGlobalXTCAVCalibration(evt)
            saturation_value = xtup.GetCameraSaturationValue(evt)

            if not roi_xtcav or not global_calibration or not saturation_value:
                continue

            return roi_xtcav, global_calibration, saturation_value, t

        return roi_xtcav, global_calibration, saturation_value, end_of_images

LasingOffParameters = xtu.namedtuple('LasingOffParameters', 
    ['experiment', 
    'maxshots', 
    'run', 
    'start',
    'darkreferencepath', 
    'num_bunches', 
    'num_groups', 
    'snrfilter', 
    'roiwaistthres', 
    'roiexpand', 
    'islandsplitmethod',
    'islandsplitpar1', 
    'islandsplitpar2', 
    'calpath', 
    'version'])
