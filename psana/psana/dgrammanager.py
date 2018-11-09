import sys, os
import time
import getopt
import pprint

from psana import dgram
from psana.event import Event
from psana.detector import detectors
import numpy as np

def dumpDict(dict,indent):
    for k in sorted(dict.keys()):
        if hasattr(dict[k],'__dict__'):
            print(' '*indent,k)
            dumpDict(dict[k].__dict__,indent+2)
        else:
            print(' '*indent,k,dict[k])

# method to dump dgrams to stdout.  ideally this would move into dgram.cc
def dumpDgram(d):
    dumpDict(d.__dict__,0)

FN_L = 200

class DgramManager():
    
    def __init__(self, xtc_files, configs=[]):
        """ Opens xtc_files and stores configs."""
        if isinstance(xtc_files, (str)):
            self.xtc_files = np.array([xtc_files], dtype='U%s'%FN_L)
        elif isinstance(xtc_files, (list, np.ndarray)):
            self.xtc_files = np.asarray(xtc_files, dtype='U%s'%FN_L)
        assert len(self.xtc_files) > 0
        
        given_configs = True if len(configs) > 0 else False
        
        self.configs = []
        if given_configs: 
            self.configs = configs
            for i in range(len(self.configs)): 
                self.configs[i]._assign_dict()
        
        self.fds = []
        for i, xtcdata_filename in enumerate(self.xtc_files):
            self.fds.append(os.open(xtcdata_filename,
                            os.O_RDONLY))
            if not given_configs: 
                d = dgram.Dgram(file_descriptor=self.fds[-1])
                self.configs += [d]

        self.offsets = [_config._offset for _config in self.configs]
        self.det_class_table = self.get_det_class_table()
   
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def next(self, offsets=[], sizes=[], read_chunk=True):
        assert len(self.offsets) > 0 or len(offsets) > 0
        
        if len(offsets) == 0: offsets = self.offsets
        if len(sizes) == 0: sizes = [0]*len(offsets)

        dgrams = []
        for fd, config, offset, size in zip(self.fds, self.configs, offsets, sizes):
            if (read_chunk) :
                d = dgram.Dgram(config=config, offset=offset)
            else:
                assert size > 0
                d = dgram.Dgram(file_descriptor=fd, config=config, offset=offset, size=size)   
            dgrams += [d]
        
        evt = Event(dgrams, self.det_class_table)
        self.offsets = evt._offsets
        return evt

    def jump(self, offsets, sizes):
        """ Jumps to the offset and reads out dgram on each xtc file.
        This is used in normal mode (multiple detectors with MPI).
        """
        assert len(offsets) > 0 and len(sizes) > 0
        dgrams = []
        for fd, config, offset, size in zip(self.fds, self.configs, offsets, sizes):
            d = dgram.Dgram(file_descriptor=fd, config=config, offset=offset, size=size)   
        dgrams += [d]
        
        evt = Event(dgrams, self.det_class_table)
        return evt

    def get_det_class_table(self):
        """
        this function gets the version number for a (det, drp_class) combo
        maps (dettype,software,version) to associated python class
        """

        det_class_table = {}

        # loop over the dgrams in the configuration
        # if a detector/drp_class combo exists in two cfg dgrams
        # it will be OK... they should give the same final Detector class
        for cfg_dgram in self.configs:
            for det_name, det in cfg_dgram.software.__dict__.items():
                for drp_class_name, drp_class in det.__dict__.items():

                    # FIXME: we want to skip '_'-prefixed drp_classes
                    #        but this needs to be fixed upstream
                    if drp_class_name in ['dettype', 'detid']: continue
                    if drp_class_name.startswith('_'): continue

                    # use this info to look up the desired Detector class
                    versionstring = [str(v) for v in drp_class.version]
                    class_name = '_'.join([det.dettype, drp_class.software] + versionstring)
                    if hasattr(detectors, class_name):
                        DetectorClass = getattr(detectors, class_name) # return the class object
                        # TODO: implement policy for picking up correct det implementation
                        #       given the version number
                        det_class_table[(det_name, drp_class_name)] = DetectorClass
                    else:
                        pass

        return det_class_table


def parse_command_line():
    opts, args_proper = getopt.getopt(sys.argv[1:], 'hvd:f:')
    xtcdata_filename="data.xtc"
    for option, parameter in opts:
        if option=='-h': usage_error()
        if option=='-f': xtcdata_filename = parameter
    if xtcdata_filename is None:
        xtcdata_filename="data.xtc"
    return (args_proper, xtcdata_filename)

def getMemUsage():
    pid=os.getpid()
    ppid=os.getppid()
    cmd="/usr/bin/ps -q %d --no-headers -eo size" % pid
    p=os.popen(cmd)
    size=int(p.read())
    return size

def main():
    args_proper, xtcdata_filename = parse_command_line()
    ds=DgramManager(xtcdata_filename)
    print("vars(ds):")
    for var_name in sorted(vars(ds)):
        print("  %s:" % var_name)
        e=getattr(ds, var_name)
        if not isinstance(e, (tuple, list, int, float, str)):
            for key in sorted(e.__dict__.keys()):
                print("%s: %s" % (key, e.__dict__[key]))
    print()
    count=0
    for evt in ds:
        print("evt:", count)
        for dgram in evt:
            for var_name in sorted(vars(dgram)):
                val=getattr(dgram, var_name)
                print("  %s: %s" % (var_name, type(val)))
            a=dgram.xpphsd.raw.array0Pgp
            try:
                a[0][0]=999
            except ValueError:
                print("The dgram.xpphsd.raw.array0Pgp is read-only, as it should be.")
            else:
                print("Warning: the evt.array0_pgp array is writable")
            print()
        count+=1
    return

def usage_error():
    s="usage: python %s" %  os.path.basename(sys.argv[0])
    sys.stdout.write("%s [-h]\n" % s)
    sys.stdout.write("%s [-f xtcdata_filename]\n" % (" "*len(s)))
    sys.exit(1)

if __name__=='__main__':
    main()

