from psana.eventbuilder import EventBuilder
from psana.psexp.packet_footer import PacketFooter

from .callback_batch import CallbackBatchBuilder
from .run import RunSmallData


class EventBuilderManager(object):
    def __init__(self, view, configs, dsparms, callback_run_state=None):
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
                               use_proxy_events=use_proxy_events,
                               gpu_stream_ids=getattr(dsparms, 'gpu_stream_ids', None))
        self.run_smd = RunSmallData(
            self.eb,
            configs,
            dsparms,
            callback_run_state=callback_run_state,
        )  # only used by smalldata callback
        self.callback_batch_builder = CallbackBatchBuilder(
            self.eb,
            self.run_smd,
            dsparms.smd_callback,
            batch_size=dsparms.batch_size,
            respect_batch_size=True,
        )

    def _normalize_batch_result(self, batch_result):
        if len(batch_result) == 3:
            return batch_result

        batch_dict, step_dict = batch_result
        return batch_dict, {}, step_dict

    def batches_with_gpu(self):
        while True:
            # This either calls user-defined smalldata callback, which loops
            # over smd events or skips (faster). To enable detector interface
            # for smd events, evt.complete() (slow) is called.
            # Note: use _smd_callback for checking if user set any callback
            # through DataSource.
            if self.dsparms.smd_callback == 0:
                batch_result = self.eb.build()
                if self.eb.nevents == 0 and self.eb.nsteps == 0:
                    break
                batch_dict, gpu_batch_dict, step_dict = self._normalize_batch_result(
                    batch_result
                )
            else:
                callback_batch = self.callback_batch_builder.next_batch()
                if callback_batch is None:
                    break
                batch_dict, step_dict = callback_batch
                gpu_batch_dict = {}

            yield batch_dict, gpu_batch_dict, step_dict

    def batches(self):
        for batch_dict, _, step_dict in self.batches_with_gpu():
            yield batch_dict, step_dict
