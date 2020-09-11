from psana.eventbuilder import EventBuilder
from psana.psexp        import PacketFooter, PrometheusManager

class EventBuilderManager(object):

    def __init__(self, view, run): 
        self.configs        = run.configs 
        self.batch_size     = run.batch_size
        self.filter_fn      = run.filter_callback
        self.destination    = run.destination
        self.n_files        = len(self.configs)

        pf                  = PacketFooter(view=view)
        views               = pf.split_packets()
        self.eb             = EventBuilder(views, self.configs)
        self.c_filter       = PrometheusManager.get_metric('psana_eb_filter')

    def batches(self):
        while True: 
            batch_dict, step_dict = self.eb.build(
                    batch_size          = self.batch_size, 
                    filter_fn           = self.filter_fn, 
                    destination         = self.destination,
                    prometheus_counter  = self.c_filter)
            self.min_ts = self.eb.min_ts
            self.max_ts = self.eb.max_ts
            print(f"call build self.eb.nevents={self.eb.nevents} nsteps={self.eb.nsteps}")
            if self.eb.nevents==0 and self.eb.nsteps==0: break
            yield batch_dict, step_dict

