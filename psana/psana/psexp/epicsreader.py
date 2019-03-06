import sys, os, fcntl
from threading import Thread
from psana import dgram
import numpy as np

BLOCKSIZE = 0x100000

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty # python 2.x

def producer(fd, queue):
    for buf in nonblocking_read(fd):
        queue.put(buf)

def nonblocking_read(fd):
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    
    while True:
        try:
            block = os.read(fd, BLOCKSIZE)
        except BlockingIOError:
            yield bytearray()  #yield empty buffer if no data
            continue

        if not block:
            yield bytearray()
            break

        yield block
        
def consumer(queue):
    while True:
        try: 
            block = queue.get_nowait() # or q.get(timeout=.1)
        except Empty:
            yield bytearray()
            break
        else:
            yield block

class EpicsReader(object):
    """ Reads block of Epics data
    The reader uses two threads: producer continuously reads out
    block of data from epics xtc and put each block in a queue 
    and consumer returns accumulated blocks of data as presented
    in the queue."""
    epics_fd = -1
    epics_config = None
    
    def __init__(self, epics_file):
        """ Activates the reading thread that continously putting
        chunks of data to the queue. For experiments with no Epics file, 
        the reading thread will not be activated and queue will be empty.""" 
        self.queue = Queue()
        if epics_file:
            self.epics_fd = os.open(epics_file, os.O_RDONLY)
            self.epics_config = dgram.Dgram(file_descriptor=self.epics_fd)
            t1 = Thread(target=producer, args=(self.epics_fd, self.queue))
            t1.daemon = True
            t1.start()
        else:
            self.epics_config = bytearray()

    def read(self):
        """ Packs all blocks in the queue and return. """
        buf = bytearray()
        for block in consumer(self.queue):
            buf.extend(block)
        return buf

    @property
    def _config(self):
        return self.epics_config

if __name__ == "__main__":
    epics_file = "/reg/d/psdm/xpp/xpptut15/scratch/mona/xtc2/data-r0001-epc.xtc2"
    epics_reader = EpicsReader(epics_file)
    for buf in epics_reader.read():
        print('received %d bytes'%(len(buf)))
