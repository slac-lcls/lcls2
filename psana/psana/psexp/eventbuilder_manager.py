from psana.eventbuilder import EventBuilder
from psana.psexp        import PacketFooter, PrometheusManager

class EventBuilderManager(object):

    def __init__(self, view, configs, dsparms, run): 
        self.configs        = configs 
        self.n_files        = len(self.configs)
        c_filter            = PrometheusManager.get_metric('psana_eb_filter')

        pf                  = PacketFooter(view=view)
        views               = pf.split_packets()
        self.eb             = EventBuilder(views, self.configs, dsparms.timestamps,
                                           batch_size=dsparms.batch_size,
                                           intg_stream_id=dsparms.intg_stream_id,
                                           filter_fn=dsparms.filter,
                                           destination=dsparms.destination,
                                           run=run,
                                           prometheus_counter=c_filter)

    def batches(self):
        while True: 
            batch_dict, step_dict = self.eb.build()
            self.min_ts = self.eb.min_ts
            self.max_ts = self.eb.max_ts
            if self.eb.nevents==0 and self.eb.nsteps==0: break
            yield batch_dict, step_dict

