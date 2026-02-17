from psana.eventbuilder import EventBuilder
from psana.psexp.packet_footer import PacketFooter

from .callback_batch import CallbackBatchBuilder
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
        self.callback_batch_builder = CallbackBatchBuilder(
            self.eb,
            self.run_smd,
            dsparms.smd_callback,
            batch_size=dsparms.batch_size,
            respect_batch_size=True,
        )

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
                callback_batch = self.callback_batch_builder.next_batch()
                if callback_batch is None:
                    break
                batch_dict, step_dict = callback_batch

            yield batch_dict, step_dict
