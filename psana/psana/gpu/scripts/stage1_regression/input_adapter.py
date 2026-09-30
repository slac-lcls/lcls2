"""Benchmark-only dense input request for pre/post Stage 1b runtimes.

Both variants start without calibration dictionaries. Install AFTER legacy MPI
IPC setup has seen an empty detector map, so no constants or geometry are made.
The old producer's process_batch shim queues exactly one prepare_batch and
publishes nothing. Its slot-owned arrays survive until ordinary pool retirement.
"""
from common import digest


def install(checks, pixels):
    from psana.gpu.gpu_detector import DenseInputPreparer
    from psana.gpu.gpu_input import GpuDetectorBinding
    from psana.psexp.mpi_ds import RunParallel

    class RequestedInput(DenseInputPreparer):
        _passthrough = True  # old transition dispatcher: no calibration refresh

        def prepare_batch(self, *args, **kwargs):
            prepared = super().prepare_batch(*args, **kwargs)
            if prepared is not None and pixels:
                # Diagnostic processes only: validate the actual gathered rows.
                stream = kwargs.get('stream')
                for i, event in enumerate(prepared.events):
                    if event.timestamp in pixels:
                        if stream is not None:
                            stream.synchronize()
                        assert prepared.present[i].get().all()
                        checks.append(dict(timestamp=event.timestamp,
                                           raw=digest(prepared.data[i].get())))
            return prepared

        def process_batch(self, *args, **kwargs):
            self.prepare_batch(*args, **kwargs)
            return ()

        def memory_bytes(self):
            return dict(constants=0, geometry=0, calib_slots=0,
                        **super().memory_bytes())

    make = RunParallel._make_gpu_event_manager

    def requested(run):
        manager = make(run)
        binding = manager.gpu_detector_bindings['jungfrau']
        binding = GpuDetectorBinding('jungfrau',
            canonical_segment_ids=binding.canonical_segment_ids,
            field_handles_by_segment=binding.field('raw', 'raw').field_handles_by_segment,
            field_handles_by_name={key: field.field_handles_by_segment
                                   for key, field in binding.fields.items()})
        prep = RequestedInput.jungfrau_raw(manager.gpu_xtc_configs, binding,
            n_slots=manager.event_pool.depth, budget=manager._gpu_budget)
        prep.configure_gather(manager.gpu_xtc_parser.handle_indices)
        manager.gpu_detector_bindings['jungfrau'] = binding
        if hasattr(manager, 'input_preparers'):
            manager.input_preparers['jungfrau'] = prep
        else:
            assert manager.gpu_detectors == {}
            manager.gpu_detectors['jungfrau'] = (None, prep)
        manager._subbatch_budget_bytes = manager._compute_subbatch_budget()
        return manager

    RunParallel._make_gpu_event_manager = requested
