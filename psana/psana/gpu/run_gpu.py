import time

from psana import utils
from psana.psexp import TransitionId
from psana.psexp.events import Events
from psana.psexp.run import Run

from psana.gpu.runtime import make_gpu_runtime
from psana.gpu.profiler import GpuProfiler
from psana.gpu.record_stream import iter_records


class RunGpu(Run):
    """GPU-enabled Run path.

    Patch 5 makes the run generic by delegating detector-specific work to a
    backend object selected from ``gpu_detectors``. The pipeline still runs as a
    no-op skeleton, but the orchestration no longer needs detector-specific
    knowledge.
    """

    def __init__(
        self,
        expt,
        runnum,
        timestamp,
        dsparms,
        dm,
        smdr_man,
        configs,
        begingrun_dgrams,
        comms=None,
    ):
        init_start = time.perf_counter()
        super().__init__(expt, runnum, timestamp, dsparms, dm, smdr_man, begingrun_dgrams)
        self.configs = configs
        self.comms = comms
        self._evt_iter = None
        self._nodetype = None
        self.profiler = GpuProfiler.from_dsparms(
            self.dsparms,
            logger=self.logger,
            run_label=f"{self.expt}:r{self.runnum}",
        )
        self.runtime = make_gpu_runtime(run=self, profiler=self.profiler)
        # Keep compatibility aliases for code that still reaches into RunGpu internals.
        self.backend = getattr(self.runtime, 'backend', None)
        self.pipeline = getattr(self.runtime, 'pipeline', None)

        super()._setup_envstore()

        if self.comms is None:
            self._evt_iter = Events(
                configs,
                dm,
                self.dsparms.max_retries,
                self.dsparms.use_smds,
                self.shared_state,
                smdr_man=smdr_man,
            )
        else:
            self._setup_mpi_nodes(dm, smdr_man, configs)

        self._setup_run_calibconst()
        self.profiler.emit_summary = (self.comms is None or self._nodetype == "bd")
        self.profiler.record_initialization(time.perf_counter() - init_start)

    def _setup_mpi_nodes(self, dm, smdr_man, configs):
        # Import lazily to avoid a serial-import cycle with mpi_ds.
        from psana.psexp import mpi_ds as mpi_module

        self._nodetype = mpi_module.nodetype
        if self._nodetype == "smd0":
            self.smd0 = mpi_module.Smd0(self.comms, smdr_man, configs)
        elif self._nodetype == "eb":
            self.eb_node = mpi_module.EventBuilderNode(self.comms, configs, self.dsparms)
        elif self._nodetype == "bd":
            self.bd_node = mpi_module.BigDataNode(
                self.comms,
                configs,
                dm,
                self.dsparms,
                self.shared_state,
            )

    def start(self):
        if self.comms is None:
            yield from self._evt_iter
            return

        if self._nodetype == "smd0":
            self.smd0.start()
        elif self._nodetype == "eb":
            self.eb_node.start()
        elif self._nodetype == "bd":
            if getattr(self.runtime, "uses_smdonly", False):
                yield from self.bd_node.iter_smdonly()
                return
            yield from self.bd_node.start()

    def Detector(self, name, accept_missing=False, **kwargs):
        return self.runtime.make_detector(name, accept_missing=accept_missing, **kwargs)

    def events(self):
        if getattr(self.runtime, "uses_smdonly", False):
            raise NotImplementedError(
                "bulk-reader runtime is not yet wired into run.events(); use bulk_reader_events() or bulk_reader_batches() during the experimental phase"
            )
        loop_start = time.perf_counter()
        try:
            for rec in iter_records(self.start(), self._run_ctx):
                if rec.is_transition:
                    yield from self.runtime.handle_transition(rec)
                    self._handle_transition(rec.dgrams)
                    if rec.service == TransitionId.EndRun:
                        return
                    continue

                while not self.runtime.has_free_slot():
                    yield from self.runtime.wait_ready()

                self.runtime.submit_l1(rec)
                yield from self.runtime.pop_ready()
            yield from self.runtime.flush()
        finally:
            self.runtime.drain()
            self.profiler.record_event_loop_wall(time.perf_counter() - loop_start)
            self.runtime.finalize()

    def bulk_reader_batches(self):
        if not getattr(self.runtime, "uses_smdonly", False):
            raise RuntimeError("bulk_reader_batches() is only available for the bulk-reader runtime")

        for smd_dgrams in self.start():
            service = utils.first_service(smd_dgrams)
            timestamp = utils.first_timestamp(smd_dgrams)
            if not TransitionId.isEvent(service):
                continue
            batch, ingress = self.runtime.ingest_smd_dgrams(
                smd_dgrams=smd_dgrams,
                service=service,
                timestamp=timestamp,
            )
            if batch.is_empty:
                continue
            yield batch, ingress

    def bulk_reader_parsed_batches(self):
        if not getattr(self.runtime, "uses_smdonly", False):
            raise RuntimeError("bulk_reader_parsed_batches() is only available for the bulk-reader runtime")

        for smd_dgrams in self.start():
            service = utils.first_service(smd_dgrams)
            timestamp = utils.first_timestamp(smd_dgrams)
            if not TransitionId.isEvent(service):
                continue
            parsed = self.runtime.ingest_and_parse_smd_dgrams(
                smd_dgrams=smd_dgrams,
                service=service,
                timestamp=timestamp,
            )
            if parsed is None:
                continue
            yield parsed

    def bulk_reader_events(self):
        for parsed in self.bulk_reader_parsed_batches():
            yield parsed.event
