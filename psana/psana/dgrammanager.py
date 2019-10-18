import sys, os
import time
import getopt
import pprint

from shmem import PyShmemClient
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
    
    def __init__(self, xtc_files, configs=[], tag=None):
        """ Opens xtc_files and stores configs."""
        self.xtc_files = []
        self.shmem = None
        self.shmem_kwargs = {'index':-1,'size':0,'cli':None}
        self.configs = []
        self.fds = []
        self._timestamps = [] # built when iterating 

        if isinstance(xtc_files, (str)):
            self.xtc_files = np.array([xtc_files], dtype='U%s'%FN_L)
            assert len(self.xtc_files) > 0
        elif isinstance(xtc_files, (list, np.ndarray)):
            if len(xtc_files) > 0: # handles smalldata-only case
                if xtc_files[0] == 'shmem':
                    self.shmem = PyShmemClient()
                    #establish connection to available server - blocking
                    status = int(self.shmem.connect(tag,0))
                    assert not status,'shmem connect failure %d' % status
                    #wait for first configure datagram - blocking
                    view = self.shmem.get(self.shmem_kwargs)
                    assert view
                    d = dgram.Dgram(view=view, \
                                    shmem_index=self.shmem_kwargs['index'], \
                                    shmem_size=self.shmem_kwargs['size'], \
                                    shmem_cli=self.shmem_kwargs['cli'])
                    self.configs += [d]
                else:    
                    self.xtc_files = np.asarray(xtc_files, dtype='U%s'%FN_L)
                    assert len(self.xtc_files) > 0
            
        given_configs = True if len(configs) > 0 else False
        
        if given_configs: 
            self.configs = configs
        
        for i, xtcdata_filename in enumerate(self.xtc_files):
            self.fds.append(os.open(xtcdata_filename,
                            os.O_RDONLY))
            if not given_configs: 
                d = dgram.Dgram(file_descriptor=self.fds[-1])
                self.configs += [d]

        self.det_class_table, self.xtc_info = self.get_det_class_table()
        self.calibs = {} # initialize to empty dict - will be populated by run class

    def __del__(self):
        if self.fds:
            for fd in self.fds:
                os.close(fd)

    def __iter__(self):
        return self

    def __next__(self):
#        return self.next()
#    
#    def next(self):
        """ only support sequential read - no event building"""
        if self.shmem:
            view = self.shmem.get(self.shmem_kwargs)
            if view:
                # use the most recent configure datagram
                config = self.configs[len(self.configs)-1]
                d = dgram.Dgram(config=config,view=view, \
                                shmem_index=self.shmem_kwargs['index'], \
                                shmem_size=self.shmem_kwargs['size'], \
                                shmem_cli=self.shmem_kwargs['cli'])
                dgrams = [d]
            else:
                raise StopIteration
        else:
            dgrams = [dgram.Dgram(config=config) for config in self.configs]
 
        evt = Event(dgrams)
        self._timestamps += [evt.timestamp]
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
        
        evt = Event(dgrams)
        return evt

    def get_det_class_table(self):
        """
        this function gets the version number for a (det, drp_class) combo
        maps (dettype,software,version) to associated python class
        """

        det_class_table = {}
        xtc_info = []

        # loop over the dgrams in the configuration
        # if a detector/drp_class combo exists in two cfg dgrams
        # it will be OK... they should give the same final Detector class

        for cfg_dgram in self.configs:
            for det_name, det_dict in cfg_dgram.software.__dict__.items():
                # go find the class of the first segment in the dict
                # they should all be identical
                first_key = next(iter(det_dict.keys()))
                det = det_dict[first_key]
                for drp_class_name, drp_class in det.__dict__.items():

                    # FIXME: we want to skip '_'-prefixed drp_classes
                    #        but this needs to be fixed upstream
                    if drp_class_name in ['dettype', 'detid']: continue
                    if drp_class_name.startswith('_'): continue

                    # use this info to look up the desired Detector class
                    versionstring = [str(v) for v in drp_class.version]
                    class_name = '_'.join([det.dettype, drp_class.software] + versionstring)
                    xtc_entry = (det_name,det.dettype,drp_class_name,'_'.join(versionstring))
                    if xtc_entry not in xtc_info:
                        xtc_info.append(xtc_entry)
                    if hasattr(detectors, class_name):
                        DetectorClass = getattr(detectors, class_name) # return the class object
                        det_class_table[(det_name, drp_class_name)] = DetectorClass
                    else:
                        pass

        return det_class_table,xtc_info

    def get_timestamps(self):
        return np.asarray(self._timestamps, dtype=np.uint64) # return numpy array for easy search later


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

