from psana.event import Event
from psana.psexp.packet_footer import PacketFooter
import numpy as np

class EventManager(object):

    def __init__(self, smd_configs, dm):
        self.smd_configs = smd_configs
        self.dm = dm

    def events(self, view):
        pf = PacketFooter(view=view)
        views = pf.split_packets()
        for event_bytes in views:
            if event_bytes:
                evt = Event()._from_bytes(self.smd_configs, event_bytes)
                # get big data
                ofsz = np.asarray([[d.info.offsetAlg.intOffset, d.info.offsetAlg.intDgramSize] \
                        for d in evt])
                bd_evt = self.dm.jump(ofsz[:,0], ofsz[:,1])
                yield bd_evt
