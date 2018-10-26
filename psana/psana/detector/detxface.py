
from psana import DataSource
import detectors

# this routine looks in the event dgram to see what detector
# names (e.g. 'xppcspad') are present on this event, and what data
# names are present (e.g. 'raw', 'fex').  It then looks in the
# associated config dgram to see what software/version is needed
# to understand this data, and creates a list of raw dgrams
# that is usable by the Detector xface.  It is stored as a
# map so that we can append more dgrams to list of a particular
# detector as they are discovered in the raw data.

# FIXME: since this routine runs every event, performance is a concern
# TODO:  add configs & calibs
# TODO:  detectors.py : should it be heirarchical and how?
# TODO:  doc strings

# WE WANT:
# evt.xppcspad.{raw, fex, ...}
#   with the stuff in {} defined by us (in the detector interface)

# WE GET:
# evt._dgram.xppcspad.{raw, fex}                       
#                    .roi.region()                     (per-event metadata)
#                    .roi.raw()                        (possibly opaque data defined by DAQ)
# run._configs[i].xppcspad                             (configuration metadata)
# run._configs[i].software.xppcspad.{raw, fex, ...}.   (software for interpretation)


class DrpClassContainer(object):
    def __init__(self):
        pass


def add_det_xface(event, det_class_table):
    """
    """

    for evt_dgram in event.dgrams:
        for det_name, det in evt_dgram.__dict__.items():

            # this gives us the intermediate "det" level
            # in the detector interface
            if hasattr(evt, det_name):
                det_xface_obj = getattr(evt, det_name)
            else:
                det_xface_obj = DrpClassContainer()
                setattr(evt, det_name, det_xface_obj)                

            # now the final "drp_class" level
            for drp_class_name, drp_class in det.__dict__.items():

                # IF the final level detector interface object is NOT instantiated
                # THEN create the instance first
                if not hasattr(det_xface_obj, drp_class_name):
                    DetectorClass = det_class_table[(det_name, drp_class_name)]
                    detector_instance = DetectorClass()
                    setattr(det_xface_obj, drp_class_name, detector_instance)
                else:
                    detector_instance = getattr(det_xface_obj, drp_class_name)

                # and add dgram data
                detector_instance._append_dgram(drp_class)


    return



# TJ says: this function just gets the version number for a (det, drp_class) combo
# maps (dettype,software,version) to associated python class
def get_det_class_table(cfgs):
    #print( ds._configs[0].software.xppcspad.dettype )      # cspad
    #print( ds._configs[0].software.xppcspad.raw.software ) # raw, fex, ...
    #print( ds._configs[0].software.xppcspad.raw.version )  # 3.2.1

    det_class_table = {} 

    # loop over the dgrams in the configuration
    # if a detector/drp_class combo exists in two cfg dgrams
    # it will be OK... they should give the same final Detector class
    for cfg_dgram in cfgs:
        for det_name, det in cfg_dgram.software.__dict__.items():
            for drp_class_name, drp_class in det.__dict__.items():

                # FIXME: we want to skip '_'-prefixed drp_classes
                #        but this needs to be fixed upstream
                if drp_class_name in ['dettype', 'detid']:
                #if drp_class_name.startswith('_'):
                    continue

                # use this info to look up the desired Detector class
                versionstring = [str(v) for v in drp_class.version]
                class_name = '_'.join([det.dettype, drp_class.software] + versionstring)
                if hasattr(detectors, class_name):
                    DetectorClass = getattr(detectors, class_name) # return the class object
                else:
                    raise NotImplementedError(class_name)

                det_class_table[(det_name, drp_class_name)] = DetectorClass

    return det_class_table


if __name__ == '__main__':

    #ds = DataSource('data.xtc')
    ds = DataSource('/reg/neh/home/cpo/git/lcls2/hsd_oct22_tp2.xtc')
    firstTime = True

    import time

    for nevt,evt in enumerate(ds.events()):

        if firstTime:
            firstTime=False
            det_class_table = get_det_class_table(ds._configs)
            tstart = time.time()
        
        # we can discover what detector to use from dettype & raw.software
        #print( ds._configs[0].software.xppcspad.detid )        # crazy serial #
        #print( ds._configs[0].software.xppcspad.dettype )      # cspad
        #print( ds._configs[0].software.xppcspad.raw.software ) # raw, fex, ...
        #print( ds._configs[0].software.xppcspad.raw.version )  # 3.2.1

        add_det_xface(evt, det_class_table)

    print((nevt)/(time.time()-tstart))
