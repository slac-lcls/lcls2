import os

from psana.eventbuilder import EventBuilder, PlanEventBuilder
from psana.psexp.packet_footer import PacketFooter

from .run import RunSmallData

USE_PLAN_BUILDER = bool(int(os.environ.get("PS_EB_SEND_BD_PLAN", "0")))


class EventBuilderManager(object):
    def __init__(self, view, configs, dsparms, n_bd_nodes=1):
        self.configs = configs
        self.dsparms = dsparms
        self.n_files = len(self.configs)
        self.n_bd_nodes = max(1, int(n_bd_nodes))
        self.latest_plan = None

        pf = PacketFooter(view=view)
        views = pf.split_packets()
        use_proxy_events = bool(dsparms.smd_callback or getattr(dsparms, "intg_det", ""))
        plan_enabled = USE_PLAN_BUILDER and dsparms.smd_callback == 0
        builder_cls = PlanEventBuilder if plan_enabled else EventBuilder
        self._plan_enabled = plan_enabled
        builder_kwargs = dict(
            filter_timestamps=dsparms.timestamps,
            intg_stream_id=dsparms.intg_stream_id,
            batch_size=dsparms.batch_size,
            use_proxy_events=use_proxy_events,
        )
        if plan_enabled:
            builder_kwargs["n_bd_nodes"] = self.n_bd_nodes
            builder_kwargs["use_smds_map"] = getattr(dsparms, "use_smds", None)
        self.eb = builder_cls(
            views,
            self.configs,
            **builder_kwargs,
        )
        self.run_smd = RunSmallData(self.eb, configs, dsparms)  # only used by smalldata callback

    def batches(self):
        while True:
            # This eiter calls user-defined smalldata callback, which loops
            # over smd events or skips (faster). To enable detector inteface
            # for smd events, evt.complete() (slow) is called.
            # Note: use _smd_callback for checking if user set any callback
            # through DataSource.
            if self.dsparms.smd_callback == 0:
                if self._plan_enabled:
                    plan, batch_dict, step_dict = self.eb.build_with_plan()
                    self.latest_plan = plan
                else:
                    batch_dict, step_dict = self.eb.build()
                    self.latest_plan = None
                if self.eb.nevents == 0 and self.eb.nsteps == 0:
                    break
            else:
                self.latest_plan = None
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
