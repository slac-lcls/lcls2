import copy
import os
import time
import psana
import numpy as np
import glob
import sys
import getopt
import warnings
import psana.xtcav.UtilsPsana as xtup
from psana.xtcav.FileInterface import Load as constLoad
from psana.xtcav.FileInterface import Save as constSave
from psana.xtcav.CalibrationPaths import *
import psana.xtcav.Constants as Cn
from psana.xtcav.Utils import namedtuple, ROIMetrics  
"""
    Class that generates a dark background image for XTCAV reconstruction purposes. Essentially takes valid
    dark reference images and averages them to find the "average" camera background. It is recommended to use a 
    large number of shots so that spikes in energy levels can get rounded out. 
    Arguments:
        experiment (str): String with the experiment reference to use. E.g. 'amoc8114'
        run (str): String with a run number. E.g. '123' 
        max_shots (int): Maximum number of images to use for the reference.
        calibration_path (str): Custom calibration directory in case the default is not intended to be used.
        validity_range (tuple): If not set, the validity range for the reference will go from the 
        first run number used to generate the reference and the last run.
"""

class DarkBackgroundReference(object):
    def __init__(self, 
        experiment='amoc8114', 
        max_shots=401, 
        run_number='86', 
        start_image=0,
        validity_range=None, 
        calibration_path='',
        save_to_file=True):

        self.image=None
        self.ROI=None
        self.n=0

        self.parameters = DarkBackgroundParameters(
            experiment = experiment, max_shots = max_shots, run_number = run_number, 
            validity_range = validity_range, calibration_path = calibration_path)

        
        warnings.filterwarnings('always',module='Utils',category=UserWarning)
        warnings.filterwarnings('ignore',module='Utils',category=RuntimeWarning, message="invalid value encountered in divide")
        
        """
        After setting all the parameters, this method has to be called to generate the dark reference and 
        save it in the proper location. 
        """
        print('dark background reference')
        print('\t Experiment: %s' % self.parameters.experiment)
        print('\t Run: %s' % self.parameters.run_number)
        print('\t Valid shots to process: %d' % self.parameters.max_shots)
        
        #Loading the dataset from the "dark" run, this way of working should be compatible with both xtc and hdf5 files
        dataSource=psana.DataSource("exp=%s:run=%s:idx" % (self.parameters.experiment, self.parameters.run_number))
        
        #Camera and type for the xtcav images
        xtcav_camera = psana.Detector(Cn.SRC)
        
        #Stores for environment variables    
        configStore=dataSource.env().configStore()
        epicsStore=dataSource.env().epicsStore()

        n=0  #Counter for the total number of xtcav images processed 
        run = dataSource.runs().next()     
        
        roi_xtcav, first_image = self._getCalibrationValues(run, xtcav_camera, start_image)
        accumulator_xtcav = np.zeros((roi_xtcav.yN, roi_xtcav.xN), dtype=np.float64)
        
        times = run.times()  
        for t in range(first_image, len(times)):
            evt=run.event(times[t])
            img = xtcav_camera.image(evt)

            # skip if empty image
            if img is None: 
                continue
          
            accumulator_xtcav += img 
            n += 1
                
            if n % 5 == 0:
                sys.stdout.write('\r%.1f %% done, %d / %d' % (float(n) / self.parameters.max_shots*100, n, self.parameters.max_shots ))
                sys.stdout.flush()   
            if n >= self.parameters.max_shots:                    #After a certain number of shots we stop (Ideally this would be an argument, rather than a hardcoded value)
                break                          
        #At the end of the program the total accumulator is saved 
        sys.stdout.write('\nMaximum number of images processed\n') 
        self.image=accumulator_xtcav/n
        self.ROI=roi_xtcav
        
        if not self.parameters.validity_range or not type(self.parameters.validity_range) == tuple:
            self.parameters = self.parameters._replace(validity_range=(self.parameters.run_number, 'end'))
        elif len(self.parameters.validity_range) == 1:
            self.parameters = self.parameters._replace(validity_range=(self.parameters.validity_range[0], 'end'))
            
        if save_to_file:
            cp = CalibrationPaths(dataSource.env(), self.parameters.calibration_path)
            file = cp.newCalFileName(Cn.DB_FILE_NAME, self.parameters.validity_range[0], self.parameters.validity_range[1])
            self.save(file)

    
    @staticmethod
    def _getCalibrationValues(run, xtcav_camera, start_image):
        roi_xtcav = None
        times = run.times()

        end_of_images = len(times)
        for t in range(start_image,end_of_images):
            evt = run.event(times[t])
            img = xtcav_camera.image(evt)
            # skip if empty image
            if img is None: 
                continue
            roi_xtcav = xtup.getXTCAVImageROI(evt)
            
            if not roi_xtcav:
                continue

            return roi_xtcav, t

        return roi_xtcav, end_of_images


    def save(self,path): 
        instance = copy.deepcopy(self)
        if instance.ROI:
            instance.ROI = dict(vars(instance.ROI))
            instance.parameters = dict(vars(instance.parameters))
        constSave(instance,path)
        
    @staticmethod    
    def load(path):        
        obj = constLoad(path)
        try:
            obj.ROI = ROIMetrics(**obj.ROI)
            obj.parameters = DarkBackgroundParameters(**obj.parameters)
        except (AttributeError, TypeError):
            print("Could not load Dark Reference with path "+ path+". Try recreating dark reference " +\
            "to ensure compatability between versions")
            return None
        return obj

DarkBackgroundParameters = namedtuple('DarkBackgroundParameters', 
    ['experiment', 
     'max_shots', 
     'run_number', 
     'validity_range', 
     'calibration_path'])
