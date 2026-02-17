from psana.eventbuilder import EventBuilder
from psana.psexp.packet_footer import PacketFooter

from .run import RunSmallData


class EventBuilderManager(object):
    def __init__(self, view, configs, dsparms):
        self.configs = configs
        self.dsparms = dsparms
        self.n_files = len(self.configs)

        pf = PacketFooter(view=view)
        views = pf.split_packets()
        use_proxy_events = bool(dsparms.smd_callback or getattr(dsparms, "intg_det", ""))
        self.eb = EventBuilder(views,
                               self.configs,
                               filter_timestamps=dsparms.timestamps,
                               intg_stream_id=dsparms.intg_stream_id,
                               batch_size=dsparms.batch_size,
                               use_proxy_events=use_proxy_events)
        self.run_smd = RunSmallData(self.eb, configs, dsparms)  # only used by smalldata callback

    def batches(self):
        while True:
            # This eiter calls user-defined smalldata callback, which loops
            # over smd events or skips (faster). To enable detector inteface
            # for smd events, evt.complete() (slow) is called.
            # Note: use _smd_callback for checking if user set any callback
            # through DataSource.
            if self.dsparms.smd_callback == 0:
                batch_dict, step_dict = self.eb.build()
                if self.eb.nevents == 0 and self.eb.nsteps == 0:
                    break
            else:
                # Collects list of proxy events to be converted to batches.
                # Note that we are persistently calling smd_callback until there's nothing
                # left in all views used by EventBuilder. From this while/for loops, we
                # either gets transitions from SmdDataSource and/or L1 from the callback.
                while self.run_smd.proxy_events == [] and self.eb.has_more():
                    for evt in self.dsparms.smd_callback(self.run_smd):
                        self.run_smd.proxy_events.append(evt._proxy_evt)

                if not self.run_smd.proxy_events:
                    break

                # Generate a bytearray representations of all the events in a batch.
                batch_dict, step_dict = self.eb.gen_bytearray_batch(
                    self.run_smd.proxy_events
                )
                self.run_smd.proxy_events = []

            yield batch_dict, step_dict
