from psana.smdreader import SmdReader
from psana.eventbuilder import EventBuilder
from psana.event import Event
import numpy as np
import os, glob

def run_smd0(fds, max_events=0):
    n_events = int(os.environ.get('PS_SMD_N_EVENTS', 100))
    if max_events:
        if max_events < n_events:
            n_events = max_events

    smdr = SmdReader(fds)
    got_events = -1
    processed_events = 0
    while got_events != 0:
        smdr.get(n_events)
        got_events = smdr.got_events
        processed_events += got_events
        print("processed_events", processed_events, "got_events", got_events)
        views = bytearray()
        for i in range(len(fds)):
            view = smdr.view(i)
            if view != 0:
                views.extend(view)
                if i < len(fds) - 1:
                    views.extend(b'endofstream')
            
            if views:
                #run_smd_task(views, run)
                print("got views")
            
            if max_events:
                if processed_events >= max_events:
                    break
    """
    while got_events != 0:
        got_events = smdr.get(n_events)
        print('got_events', got_events)
        processed_events += got_events
        views = bytearray()
        for i in range(len(fds)):
            view = smdr.view(i)
            if view != 0:
                views.extend(view)
                if i < len(fds) - 1:
                    views.extend(b'endofstream')
            
            if views:
                run_smd_task(views, run)
            
            if run.max_events:
                if processed_events >= run.max_events:
                    break
    """

if __name__ == "__main__":
    filenames = glob.glob('.tmp/smalldata/*r0001*.xtc')
    fds = [os.open(filename, os.O_RDONLY) for filename in filenames]
    run_smd0(fds)

