from psana.eventbuilder import EventBuilder
from psana.psexp.packet_footer import PacketFooter

class EventBuilderManager(object):

    def __init__(self, view, configs, batch_size=1, filter_fn=0, destination=0):
        self.configs = configs
        self.batch_size = batch_size
        self.filter_fn = filter_fn
        self.destination = destination

        pf = PacketFooter(view=view)
        views = pf.split_packets()
        self.eb = EventBuilder(views)

    def batches(self, limit_ts=-1):
        batch = self.eb.build(batch_size=self.batch_size, filter_fn=self.filter_fn, destination=self.destination, limit_ts=limit_ts)
        while self.eb.nevents:
            self.min_ts = self.eb.min_ts
            self.max_ts = self.eb.max_ts
            yield batch
            batch = self.eb.build(batch_size=self.batch_size, filter_fn=self.filter_fn, destination=self.destination, limit_ts=limit_ts)

