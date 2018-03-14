import sys, os
import time
import getopt
import pprint

from psana import dgram
from psana.event import Event
import numpy as np

class container:
    pass

def setnames(d):  
    keys = sorted(d.__dict__.keys())
    for k in keys:
        fields = k.split('_')
        currobj = d
        for f in fields[:-1]:
            if not hasattr(currobj, f):
                setattr(currobj, f, container())
            currobj = getattr(currobj, f)
        val = getattr(d, k)
        setattr(currobj, fields[-1], val)
        delattr(d, k)

class DgramManager:
    """Stores variables and arrays loaded from an XTC source.\n"""
    def __init__(self, xtc_files, configs=[]):
        if isinstance(xtc_files, (str)):
            self.xtc_files = np.array([xtc_files], dtype='U25')
        elif isinstance(xtc_files, (list, np.ndarray)):
            self.xtc_files = np.asarray(xtc_files, dtype='U25')
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
                            os.O_RDONLY|os.O_LARGEFILE))
            if not given_configs: 
                d = dgram.Dgram(file_descriptor=self.fds[-1])
                setnames(d)
                self.configs += [d]
        
    def __iter__(self):
        return self

    def __next__(self):
        evt = self.next()
        return evt

    def next(self, offsets=[]):  
        dgrams = []
        if len(offsets) == 0: offsets = [0]*len(self.fds)
        for fd, config, offset in zip(self.fds, self.configs, offsets):
            d = dgram.Dgram(file_descriptor=fd, config=config, offset=offset)   
            setnames(d)
            dgrams += [d]
        
        evt = Event(dgrams=dgrams)
        return evt

    
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

