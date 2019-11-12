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
        batch = self.eb.build(batch_size=self.batch_size, filter_fn=self.filter_fn, destination=self.destination)
        while self.eb.nevents:
            self.min_ts = self.eb.min_ts
            self.max_ts = self.eb.max_ts
            yield batch
            batch = self.eb.build(batch_size=self.batch_size, filter_fn=self.filter_fn, destination=self.destination)

    def step_chunk(self):
        """ Returns list of steps in all smd files."""
        step_view = bytearray()
        step_pf = PacketFooter(n_packets=self.n_files)
        
        for i in range(self.n_files):
            _step_view = self.eb.step_view(i)
            if _step_view != 0:
                step_view.extend(_step_view)
                step_pf.set_size(i, memoryview(_step_view).shape[0])

        if step_view:
            step_view.extend(step_pf.footer)

        return step_view
