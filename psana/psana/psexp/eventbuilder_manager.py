from psana.eventbuilder import EventBuilder
from psana.psexp.packet_footer import PacketFooter

class EventBuilderManager(object):

    def __init__(self, configs, batch_size, filter_fn):
        self.configs = configs
        self.batch_size = batch_size
        self.filter_fn = filter_fn

    def batches(self, view):
        pf = PacketFooter(view=view)
        views = pf.split_packets()
        eb = EventBuilder(views)
        batch = eb.build(batch_size=self.batch_size, filter_fn=self.filter_fn)
        while eb.nevents:
            yield batch
            batch = eb.build(batch_size=self.batch_size, filter_fn=self.filter_fn)

