import os, time, glob, sys
import pstats, cProfile
import pyximport
pyximport.install()
from psana.smdreader import SmdReader

def run_smd0():
    #filenames = glob.glob('/reg/d/psdm/xpp/xpptut15/scratch/mona/test/smalldata/*.smd.xtc2')
    #filenames = glob.glob('/u1/mona/smalldata/*.smd.xtc2')
    filenames = glob.glob('.tmp/smalldata/*r0001*.xtc2')
    fds = [os.open(filename, os.O_RDONLY) for filename in filenames]
    limit = int(sys.argv[1])
    st = time.time()
    smdr = SmdReader(fds[:limit])
    got_events = -1
    n_events = 1000
    processed_events = 0
    while got_events != 0:
        smdr.get(n_events)
        got_events = smdr.got_events
        processed_events += got_events
    #print("processed_events: %d"%processed_events)
    en = time.time()
    print("Elapsed Time (s): %f Rate: %f"%((en-st), processed_events/((en-st)*1e6)))

if __name__ == "__main__":
    run_smd0()
    #cProfile.runctx("run_smd0()", globals(), locals(), "Profile.prof")
    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats()
