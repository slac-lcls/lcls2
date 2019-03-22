import os
from psana import dgram
import numpy as np
from psana.parallelreader import ParallelReader
from psana.psexp.packet_footer import PacketFooter

class EpicsReader(object):
    """ Reads block of Epics data
    The reader uses two threads: producer continuously reads out
    block of data from epics xtc and put each block in a queue 
    and consumer returns accumulated blocks of data as presented
    in the queue."""
    
    def __init__(self, fds):
        """ Stores configs and file descriptors"""
        if fds:
            self._prl_reader = ParallelReader(fds)
        else:
            self._prl_reader = None

    def read(self):
        """ Returns a list of memoryviews each pointing to a chunk
        of epics data."""
        mmr_views = []
        if self._prl_reader:
            block = self._prl_reader.get_block()
            if block:
                pf = PacketFooter(view=block)
                mmr_views = pf.split_packets()
        return mmr_views


if __name__ == "__main__":
    import glob, time
    epics_files = glob.glob("/reg/d/psdm/xpp/xpptut15/scratch/mona/xtc2/data-r0001-e0*.xtc2")
    epics_reader = EpicsReader(epics_files)
    block = epics_reader.read()
    while block:
        print('received block of %d bytes'%(memoryview(block).shape[0]))
        block = epics_reader.read()
