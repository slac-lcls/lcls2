
import logging
logger = logging.getLogger(__name__)

import warnings

import os
import sys
import time
from psana import DataSource
import numpy as np

import psana.xtcav.Utils as xtu
import psana.xtcav.UtilsPsana as xtup
import psana.xtcav.SplittingUtils as su
import psana.xtcav.Constants as cons
from   psana.xtcav.CalibrationPaths import *
from   psana.xtcav.DarkBackgroundReference import *
from   psana.xtcav.FileInterface import Load as constLoad

from   psana.xtcav.FileInterface import Save as constSave
#from psana.pscalib.calib.XtcavUtils import Save as constSave
#from psana.pscalib.calib.XtcavUtils import dict_from_xtcav_calib_object

from psana.pyalgos.generic.NDArrUtils import info_ndarr, print_ndarr

import psana.pyalgos.generic.Graphics as gr

#from psana.xtcav.Simulators import\
  #SimulatorEBeam,\
  #SimulatorGasDetector,\
  #SimulatorEventId,\
  #SimulatorEnvironment

## PP imports
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()
##print('Core %s ... ready' % (rank + 1)) # useful for debugging purposes
##sys.stdout.flush()

rank = 0
size = 1

"""
    Class that generates a set of lasing off references for XTCAV reconstruction purposes
    Attributes:
        experiment (str): String with the experiment reference to use. E.g. 'amox23616'
        runs (str): String with a run number, or a run interval. E.g. '131'  '134-156' 145,136'
        max_shots (int): Maximum number of images to use for the references.
        start_image (int): image in run to start from
        validity_range (tuple): If not set, the validity range for the reference will go from the 
        first run number used to generate the reference and the last run.
        calibration_path (str): Custom calibration directory in case the default is not intended to be used.
        num_bunches (int): Number of bunches.
        snr_filter (float): Number of sigmas for the noise threshold.
        num_groups (int): Number of profiles to average together for each reference.
        roi_expand (float): number of waists that the region of interest around will span around the center of the trace.
        roi_fraction (float): fraction of pixels that must be non-zero in roi(s) of image for analysis
        island_split_method (str): island splitting algorithm. Set to 'scipylabel' or 'contourLabel'  The defaults parameter is 'scipylabel'.
"""

class LasingOffReference():

    def __init__(self,
            fname='/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0131-e000200-xtcav-v2.xtc2',
            experiment='amox23616',   #Experiment label
            max_shots=401,            #Maximum number of valid shots to process
            run_number=131,           #Run number
            start_image=0,            #Starting image in run
            validity_range=None,
            dark_reference_path=None, #Dark reference information
            num_bunches=1,            #Number of bunches
            num_groups=None,          #Number of profiles to average together
            snr_filter=10,            #Number of sigmas for the noise threshold
            roi_expand=1,             #Parameter for the roi location
            roi_fraction=cons.ROI_PIXEL_FRACTION,
            island_split_method = cons.DEFAULT_SPLIT_METHOD, #Method for island splitting
            island_split_par1 = 3.0,  #Ratio between number of pixels between largest and second largest groups when calling scipy.label
            island_split_par2 = 5.,   #Ratio between number of pixels between second/third largest groups when calling scipy.label
            calibration_path='',
            save_to_file=True):

        PLOT_IMAGE = False

        if PLOT_IMAGE :
            self.fig, self.axim, self.axcb = gr.fig_img_cbar_axes(fig=None,\
            win_axim=(0.05,  0.05, 0.87, 0.93),\
            win_axcb=(0.923, 0.05, 0.02, 0.93)) #, **kwargs)

        #fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s' # '%(message)s'
        fmt='[%(levelname).1s] L%(lineno)04d : %(message)s'
        logging.basicConfig(format=fmt, datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)

        #if type(run_number) == int:
        #    run_number = str(run_number)

        self.parameters = LasingOffParameters(experiment = experiment,
            max_shots = max_shots, run_number = run_number, start_image = start_image, validity_range = validity_range, 
            dark_reference_path = dark_reference_path, num_bunches = num_bunches, num_groups=num_groups, 
            snr_filter=snr_filter, roi_expand = roi_expand, roi_fraction=roi_fraction, island_split_method=island_split_method, 
            island_split_par2 = island_split_par2, island_split_par1=island_split_par1, 
            calibration_path=calibration_path, fname=fname, version=1)

        #warnings.filterwarnings('always',module='Utils',category=UserWarning)
        #warnings.filterwarnings('ignore',module='Utils',category=RuntimeWarning, message="invalid value encountered in divide")
        
        if rank == 0:
            print('Lasing off reference')
            print('\t File name: %s' % self.parameters.fname)
            print('\t Experiment: %s' % self.parameters.experiment)
            print('\t Runs: %s' % self.parameters.run_number)
            print('\t Number of bunches: %d' % self.parameters.num_bunches)
            print('\t Valid shots to process: %d' % self.parameters.max_shots)
            print('\t Dark reference run: %s' % self.parameters.dark_reference_path)
        
        #Loading the data, this way of working should be compatible with both xtc and hdf5 files

        #ds = psana.DataSource("exp=%s:run=%s:idx" % (self.parameters.experiment, self.parameters.run_number))

        ds = DataSource(files=fname)
        run = next(ds.runs()) # run = ds.runs().next()
        #env = SimulatorEnvironment() # ds.env()

        #Camera for the xtcav images, Ebeam type, eventid, gas detectors
        camera      = run.Detector(cons.DETNAME)      # psana.Detector(cons.DETNAME)
        ebeam       = run.Detector(cons.EBEAM)        #SimulatorEBeam() # psana.Detector(cons.EBEAM)
        eventid     = run.Detector(cons.EVENTID)      #SimulatorEventId() # evt.get(psana.EventId)
        gasdetector = run.Detector(cons.GAS_DETECTOR) #SimulatorGasDetector() # psana.Detector(cons.GAS_DETECTOR)
        xtcavpars   = run.Detector(cons.XTCAVPARS)

        # Empty list for the statistics obtained from each image, the shot to shot properties,
        # and the ROI of each image (although this ROI is initially the same for each shot,
        # it becomes different when the image is cropped around the trace)
        list_image_profiles = []

        #dark_background = self._getDarkBackground(env)
        dark_data, dark_meta = camera.calibconst.get('pedestals')

        logger.debug('==== dark_meta:\n%s' % str(dark_meta))

        dark_background = xtu.xtcav_calib_object_from_dict(dark_data)
        logger.debug('==== dir(dark_background):\n%s'% str(dir(dark_background)))
        logger.debug('==== dark_background.ROI:\n%s'% str(dark_background.ROI))
        logger.debug(info_ndarr(dark_background.image, '==== dark_background.image:'))

        print('\n',100*'_','\n')

        camraw  = xtup.get_attribute(camera,      'raw')
        valsxtp = xtup.get_attribute(xtcavpars,   'valsxtp')
        valsebm = xtup.get_attribute(ebeam,       'valsebm')
        valseid = xtup.get_attribute(eventid,     'valseid')
        valsgd  = xtup.get_attribute(gasdetector, 'valsgd')

        if None in (camraw, valsxtp, valsebm, eventid, valsgd) : 
            sys.error('FATAL ERROR IN THE DETECTOR INTERFACE: MISSING ATTRIBUTE MUST BE IMPLEMENTED')

        #Calibration values needed to process images. first_event is the index of the first event with valid data
        roi_xtcav, global_calibration, saturation_value, first_event = self._getCalibrationValues(run, camraw, valsxtp, start_image)
        msg = ('_getCalibrationValues returns'\
              '\n    roi_xtcav: %s'\
              '\n    global_calibration: %s'\
              '\n    saturation_value: %s  first_event: %s')%\
              (str(roi_xtcav), str(global_calibration), str(saturation_value), str(first_event))
        #print(msg)
        logger.debug(msg)

        #times = run.times()
        #image_numbers = xtup.divideImageTasks(first_event, len(times), rank, size)

        num_processed = 0 #Counter for the total number of xtcav images processed within the run

        for nev,evt in enumerate(run.events()):
            #logger.info('Event %03d'%nev)
            img = camraw(evt)
            if img is None: continue

            #Obtain the shot to shot parameters necessary for the retrieval of the x and y axis in time and energy units
            shot_to_shot = xtup.getShotToShotParameters(evt, valsebm, valsgd, valseid)
            #logger.debug('shot_to_shot: %s' % str(shot_to_shot))

            if not shot_to_shot.valid: continue

            image_profile, _ = xtu.processImage(img, self.parameters, dark_background, global_calibration,
                                                saturation_value, roi_xtcav, shot_to_shot)

            #logger.debug(info_ndarr(image_profile, 'LasingOffReference image_profile'))

            if not image_profile:
                continue

            #Append only image profile, omit processed image
            list_image_profiles.append(image_profile)     
            num_processed += 1

            self._printProgressStatements(num_processed)

            if num_processed >= np.ceil(self.parameters.max_shots/float(size)):
                break

            if PLOT_IMAGE :

                nda = img

                mean, std = nda.mean(), nda.std()
                aran = (mean-3*std, mean+5*std)
                
                self.axim.clear()
                self.axcb.clear()
                imsh = gr.imshow(self.axim, nda, amp_range=aran, extent=None, interpolation='nearest',\
                                 aspect='auto', origin='upper', orientation='horizontal', cmap='inferno')
                cbar = gr.colorbar(self.fig, imsh, self.axcb, orientation='vertical', amp_range=aran)
                
                gr.set_win_title(self.fig, 'Event: %d' % nev)
                gr.draw_fig(self.fig)
                gr.show(mode='non-hold')

        # here gather all shots in one core, add all lists
        #image_profiles = comm.gather(list_image_profiles, root=0)
        image_profiles = list_image_profiles

        if rank != 0: return

        sys.stdout.write('\n')
        # Flatten gathered arrays
        #image_profiles = [item for sublist in image_profiles for item in sublist]

        #for i,ipf in enumerate(image_profiles) :
        #  print('XXX image_profiles %d:\n  %s'%(i,str(ipf)))

        #Since there are 12 cores it is possible that there are more references than needed. In that case we discard some
        if len(image_profiles) > self.parameters.max_shots:
            image_profiles = image_profiles[0:self.parameters.max_shots]
        
        #At the end, all the reference profiles are converted to Physical units, grouped and averaged together
        averaged_profiles = xtu.averageXTCAVProfilesGroups(image_profiles, self.parameters.num_groups);     

        self.averaged_profiles, num_groups=averaged_profiles
        self.n=num_processed
        self.parameters = self.parameters._replace(num_groups=num_groups)   

        print('ZZZ self.parameters.validity_range', self.parameters.validity_range, ' type:', type(self.parameters.validity_range))
        print('ZZZ self.parameters.run_number', self.parameters.run_number, ' type:', type(self.parameters.run_number))

        # Set validity range, replace 'end' -> 9999 othervise save does not work...
        if not self.parameters.validity_range or not type(self.parameters.validity_range) == tuple:
            self.parameters = self.parameters._replace(validity_range=(self.parameters.run_number, 9999)) # IT WAS 'end'))
        elif len(self.parameters.validity_range) == 1:
            self.parameters = self.parameters._replace(validity_range=(self.parameters.validity_range[0], 9999)) # 'end'))

        #=====================
        #sys.exit('TEST EXIT')
        #=====================

        if save_to_file:
            #cp = CalibrationPaths(env, self.parameters.calibration_path)
            #file = cp.newCalFileName(cons.LOR_FILE_NAME, self.parameters.validity_range[0], self.parameters.validity_range[1])
            fname = 'cons-%s-%04d-%s-lasingoffreference.data' % (run.expt, run.runnum, cons.DETNAME)
            self.save(fname)


    def _printProgressStatements(self, num_processed):
        # print core numb and percentage
        if num_processed % 5 == 0:
            extrainfo = '\r' if size == 1 else '\nCore %d: '%(rank + 1)
            sys.stdout.write('%s%.1f %% done, %d / %d' % (extrainfo, float(num_processed) / np.ceil(self.parameters.max_shots/float(size)) *100, num_processed, np.ceil(self.parameters.max_shots/float(size))))
            sys.stdout.flush()


#    def _getDarkBackground(self, env):
#        """
#        DEPRECATED: Internal method. Loads dark background reference
#        """
#        if not self.parameters.dark_reference_path:
#            cp = CalibrationPaths(env, self.parameters.calibration_path)
#            dark_reference_path = cp.findCalFileName(cons.DB_FILE_NAME, int(self.parameters.run_number))
#            if not dark_reference_path:
#                print ('Dark reference for run %s not found, image will not be background substracted' % self.parameters.run_number)
#                return None
#            self.parameters = self.parameters._replace(dark_reference_path = dark_reference_path)
#        return DarkBackgroundReference.load(self.parameters.dark_reference_path)


    @staticmethod
    def _getCalibrationValues(run, camraw, valsxtp, start_image):
        """
        Internal method. Sets calibration parameters for image processing
        Returns:
            roi: region of interest in image
            global_calibration: global parameters of xtcav machine
            saturation_value: value at which image is saturated and no longer valid
            first_image: index of first valid shot in run
        """
        roi_xtcav, global_calibration, saturation_value = None, None, None
        first_good_evnum = 1e6

        for nev,evt in enumerate(run.events()):
            logger.info('C-loop event %03d'%nev)
            img = camraw(evt)
            if img is None: continue

            roi_xtcav = xtup.getXTCAVImageROI(valsxtp, evt)
            logger.debug('roi_xtcav: %s' % str(roi_xtcav))

            global_calibration = xtup.getGlobalXTCAVCalibration(valsxtp, evt)
            logger.debug('global_calibration: %s' % str(global_calibration))

            saturation_value = xtup.getCameraSaturationValue(valsxtp, evt)
            logger.debug('saturation_value: %s' % str(saturation_value))

            if not roi_xtcav or not global_calibration or not saturation_value:
                continue

            first_good_evnum = nev
            break

        return roi_xtcav, global_calibration, saturation_value, first_good_evnum

# LCLS1:
#    def save(self, path):
#        ###Move this to file interface folder...
#        instance = copy.deepcopy(self)
#        instance.parameters = dict(vars(self.parameters))
#        instance.averaged_profiles = dict(vars(self.averaged_profiles))
#        constSave(instance,path)

    def save(self, path):
        instance = copy.deepcopy(self)
        instance.parameters        = dict(self.parameters._asdict())
        instance.averaged_profiles = dict(self.averaged_profiles._asdict())

        #instance = dict_from_xtcav_calib_object(instance)

        #logger.debug('XXX instance.parameters:\n%s' % str(instance.parameters))
        #logger.debug('XXX instance.__dict__:\n%s' % str(instance.__dict__))
        logger.debug('XXX self instance:\n%s' % str(instance))
        logger.debug('XXX dir(self):\n%s' % dir(self))

        constSave(instance, path)

        logger.info('%s\n\t    Saved file: %s' % (50*'_', path))
        logger.info('command to check file: hdf5explorer %s' % path)

        if True :
            d = instance.parameters
            s = 'cdb add -e %s -d %s -c lasingoffreference -r %d -f %s -i xtcav -u <user>'%\
                (d['experiment'], cons.DETNAME, d['run_number'], path)
            logger.info('command to deploy: %s' % s)

    @staticmethod
    def load(path):
        lor = constLoad(path)
        try:
            lor.parameters = LasingOffParameters(**lor.parameters)
            lor.averaged_profiles = xtu.AveragedProfiles(**lor.averaged_profiles)
        except (AttributeError, TypeError):
            print("Could not load Lasing Off Reference with path "+ path+". Try recreating lasing off " +\
            "reference to ensure compatability between versions")
            return None
        return lor

#----------

LasingOffParameters = xtu.namedtuple('LasingOffParameters', 
    ['experiment', 
    'max_shots', 
    'run_number', 
    'start_image',
    'validity_range', 
    'dark_reference_path', 
    'num_bunches', 
    'num_groups', 
    'snr_filter', 
    'roi_expand',
    'roi_fraction', 
    'island_split_method',
    'island_split_par1', 
    'island_split_par2', 
    'calibration_path', 
    'fname', 
    'version'], 
    {'num_bunches':1,                           
    'snr_filter':10,           
    'roi_expand':1,          
    'roi_fraction':cons.ROI_PIXEL_FRACTION,
    'island_split_method': cons.DEFAULT_SPLIT_METHOD})

#----------
