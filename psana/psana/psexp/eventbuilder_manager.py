from psana.eventbuilder import EventBuilder
from psana.psexp        import PacketFooter, PrometheusManager

class EventBuilderManager(object):

    def __init__(self, view, configs, dsparms, run): 
        self.configs        = configs 
        self.n_files        = len(self.configs)
        c_filter            = PrometheusManager.get_metric('psana_eb_filter')

        pf                  = PacketFooter(view=view)
        views               = pf.split_packets()
        self.eb             = EventBuilder(views, self.configs, 
                                           dsparms=dsparms,
                                           run=run,
                                           prometheus_counter=c_filter)

    def batches(self):
        while True: 
            batch_dict, step_dict = self.eb.build()
            if self.eb.nevents==0 and self.eb.nsteps==0: break
            yield batch_dict, step_dict

    # legion mode doesn't need step/transition batches-reuse build method with option to not process steps
    def smd_batches(self):
        while True:
            batch_dict = self.eb.build(False, False)
            if self.eb.nevents==0 and self.eb.nsteps==0: break
            yield batch_dict
