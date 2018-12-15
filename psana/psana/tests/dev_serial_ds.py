"""
from psana.eventbuilder import EventBuilder

views=[]
views.append(memoryview(bytearray(b'abc')))
views.append(memoryview(bytearray(b'def')))
views.append(memoryview(bytearray(b'ghi')))

ev = EventBuilder(views, None)
#ev.testme()
"""
from psana import dgram
from psana.psexp.smdreader_manager import SmdReaderManager
from psana.psexp.eventbuilder_manager import EventBuilderManager
import os, time, glob
import numpy as np
from psana.psexp.packet_footer import PacketFooter
from psana.psexp.event_manager import EventManager
from psana.dgrammanager import DgramManager

max_events = 1000

def filter(evt):
    return True

if __name__ == "__main__":
    nfiles = 16
    batch_size = 3

    smd_files = np.asarray(glob.glob('/ffb01/monarin/hsd/smalldata/*.smd.xtc'))
    xtc_files = np.asarray(glob.glob('/ffb01/monarin/hsd/*.xtc'))
    smd_dm = DgramManager(smd_files)
    smd_configs = smd_dm.configs
    dm = DgramManager(xtc_files)
    ev_man = EventManager(smd_configs, dm)
   
    #get smd chunks
    smdr_man = SmdReaderManager(smd_dm.fds, max_events)
    eb_man = EventBuilderManager(smd_configs, batch_size, filter)
    cn_d = 0
    for i, chunk in enumerate(smdr_man.chunks()):
        for j, batch in enumerate(eb_man.batches(chunk)):
            for k, evt in enumerate(ev_man.events(batch)):
                for l, d in enumerate(evt):
                    cn_d += 1

                    print('chunk %d batch %d evt %d d %d %s'%(i,j,k,l, d.xpphsd.hsd.chan00.shape))
    
    print(cn_d)
