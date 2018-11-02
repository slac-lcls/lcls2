

"""
# this routine looks in the event dgram to see what detector
# names (e.g. 'xppcspad') are present on this event, and what data
# names are present (e.g. 'raw', 'fex').  It then looks in the
# associated config dgram to see what software/version is needed
# to understand this data, and creates a list of raw dgrams
# that is usable by the Detector xface.  It is stored as a
# map so that we can append more dgrams to list of a particular
# detector as they are discovered in the raw data.


# WE WANT:
# evt.xppcspad.{raw, fex, ...}
#   with the stuff in {} defined by us (in the detector interface)

# WE GET:
# evt._dgram.xppcspad.{raw, fex}                       
#                    .roi.region()                     (per-event metadata)
#                    .roi.raw()                        (possibly opaque data defined by DAQ)
# run._configs[i].xppcspad                             (configuration metadata)
# run._configs[i].software.xppcspad.{raw, fex, ...}.   (software for interpretation)


# TODO:  add configs & calibs
# TODO:  detectors.py : should it be heirarchical and how?
# TODO:  doc strings

# ideas:
# -- policy for picking up detector implementation using data version number
"""

from psana import DataSource
import detectors


class DrpClassContainer(object):
    def __init__(self):
        pass


def add_det_xface(event, det_class_table):
    """
    """

    for evt_dgram in event._dgrams:
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
                    if (det_name, drp_class_name) in det_class_table.keys():
                        DetectorClass = det_class_table[(det_name, drp_class_name)]
                        detector_instance = DetectorClass()
                        setattr(det_xface_obj, drp_class_name, detector_instance)
                    else:
                        # detector interface implementation not found
                        pass
                else:
                    detector_instance = getattr(det_xface_obj, drp_class_name)

                # and add dgram data
                detector_instance._append_dgram(drp_class)


    return



def get_det_class_table(cfgs):
    """
    this function gets the version number for a (det, drp_class) combo
    maps (dettype,software,version) to associated python class
    """

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
                    # TODO: implement policy for picking up correct det implementation
                    #       given the version number
                    det_class_table[(det_name, drp_class_name)] = DetectorClass
                else:
                    #raise NotImplementedError(class_name)
                    #DetectorClass = DrpClassContainer
                    print('sad day for: %s' % class_name)


    return det_class_table


if __name__ == '__main__':

    ds = DataSource('data.xtc')
    #ds = DataSource('/reg/neh/home/cpo/git/lcls2/hsd_oct22_tp2.xtc')
    firstTime = True

    import time

    for nevt,evt in enumerate(ds.events()):

        if firstTime:
            firstTime=False
            det_class_table = get_det_class_table(ds._configs)
            print(det_class_table)
            tstart = time.time()
        
        add_det_xface(evt, det_class_table)

    print((nevt)/(time.time()-tstart))
