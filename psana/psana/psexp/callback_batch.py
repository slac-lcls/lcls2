from psana.psexp import TransitionId


class CallbackBatchBuilder:
    """Builds callback-driven proxy-event batches for EventBuilder."""

    def __init__(
        self,
        eb,
        run_smd,
        smd_callback,
        batch_size=0,
        respect_batch_size=False,
    ):
        self.eb = eb
        self.run_smd = run_smd
        self.smd_callback = smd_callback
        self.batch_size = batch_size
        self.respect_batch_size = respect_batch_size
        self._callback_iter = None

    def _collect_proxy_events_unbounded(self):
        # Preserve current behavior: drain one callback invocation before batching.
        while self.run_smd.proxy_events == [] and self.eb.has_more():
            for evt in self.smd_callback(self.run_smd):
                self.run_smd.proxy_events.append(evt._proxy_evt)

        if not self.run_smd.proxy_events:
            return []

        proxy_events = self.run_smd.proxy_events
        self.run_smd.proxy_events = []
        return proxy_events

    def _collect_proxy_events_bounded(self):
        # Batch callback-selected L1Accept events similarly to non-callback mode.
        max_l1 = self.batch_size if self.batch_size and self.batch_size > 0 else 0
        l1_count = 0

        while True:
            if max_l1 and l1_count >= max_l1 and self.run_smd.proxy_events:
                break

            if self._callback_iter is None:
                if not self.eb.has_more():
                    break
                self._callback_iter = iter(self.smd_callback(self.run_smd))

            try:
                evt = next(self._callback_iter)
            except StopIteration:
                self._callback_iter = None
                if self.run_smd.proxy_events:
                    break
                if not self.eb.has_more():
                    break
                continue

            proxy_evt = evt._proxy_evt
            self.run_smd.proxy_events.append(proxy_evt)
            if TransitionId.isEvent(proxy_evt.service):
                l1_count += 1
                if max_l1 and l1_count >= max_l1:
                    break

        if not self.run_smd.proxy_events:
            return []

        proxy_events = self.run_smd.proxy_events
        self.run_smd.proxy_events = []
        return proxy_events

    def next_batch(self, run_serial=False):
        if self.respect_batch_size:
            proxy_events = self._collect_proxy_events_bounded()
        else:
            proxy_events = self._collect_proxy_events_unbounded()
        if not proxy_events:
            return None
        return self.eb.gen_bytearray_batch(proxy_events, run_serial=run_serial)
