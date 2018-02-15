import sys, os
import time
import getopt
import pprint

from psana import dgram
from psana.pydgram import PyDgram
from psana.event import Event

class DataSource:
    """Stores variables and arrays loaded from an XTC source.\n"""
    def __init__(self, expStr):
        ### Parameter expStr can be an xtc filename or 
        ### an 'exp=expid:run=runno' string
        if os.path.isfile(expStr):
            self.xtcFiles = [expStr]
        else:
            ### Read xtc file list from somewhere...
            self.xtcFiles = ['e001-r0001-s00.smd.xtc','e001-r0001-s01.smd.xtc']
        self.configs = []
        self._configs = []
        assert len(self.xtcFiles) > 0
        ### Open xtc files
        for xtcdata_filename in self.xtcFiles:
            fd = os.open(xtcdata_filename,
                         os.O_RDONLY|os.O_LARGEFILE)
            self._configs.append(dgram.Dgram(file_descriptor=fd))
            self.configs.append(PyDgram(self._configs[-1]))
        
    def __iter__(self):
        return self

    def __next__(self):
        # Loop through all files and grab dgrams
        dgrams = []
        for _config in self._configs:
            d=dgram.Dgram(config=_config)
            dgrams.append(PyDgram(d))
        return Event(dgrams=dgrams)

    def jump(self, offset=0):
        # Random access only works if the datasource is
        # a single file (mostly used to jump around in bigdata).
        d=dgram.Dgram(config=self._configs[0], offset=offset)
        event = Event(dgrams=[PyDgram(d)])
        return event

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
    ds=DataSource(xtcdata_filename)
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
        for pydgram in evt:
            for var_name in sorted(vars(pydgram)):
                val=getattr(pydgram, var_name)
                print("  %s: %s" % (var_name, val))
            a=pydgram.xpphsd.raw.array0Pgp
            try:
                a[0][0]=999
            except ValueError:
                print("The pydgram.xpphsd.raw.array0Pgp is read-only, as it should be.")
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

