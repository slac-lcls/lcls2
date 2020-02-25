from psana.eventbuilder import EventBuilder
from psana.psexp.packet_footer import PacketFooter

class EventBuilderManager(object):

    def __init__(self, view, run): 
        self.configs = run.configs 
        self.batch_size = run.batch_size
        self.filter_fn = run.filter_callback
        self.destination = run.destination
        self.n_files = len(self.configs)

        pf = PacketFooter(view=view)
        views = pf.split_packets()
        self.eb = EventBuilder(views)

    def batches(self):
        batch_dict, step_dict = self.eb.build(batch_size=self.batch_size, filter_fn=self.filter_fn, destination=self.destination)
        while self.eb.nevents or self.eb.nsteps:
            self.min_ts = self.eb.min_ts
            self.max_ts = self.eb.max_ts
            yield batch_dict, step_dict
            batch_dict, step_dict = self.eb.build(batch_size=self.batch_size, filter_fn=self.filter_fn, destination=self.destination)

