"""Exercise CallbackIterator with real SMD chunks and recycled input buffers."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from psana import DataSource
from psana.eventbuilder import EventBuilder
from psana.psexp.callback_iterator import CallbackIterator
from psana.psexp.ds_base import DataSourceBase


def metadata(run, step, evt):
    motor = run.Detector("motor2")
    epics = run.Detector("HX2:DVD:GCC:01:PMON,hello1")
    return [int(evt.timestamp), motor(step) if step is not None else motor(evt), epics(evt)]


def exercise(xtc_dir, batch_size, stepnum, behavior):
    callback_calls = 0
    source_chunks = 0
    source_closed = False
    source_close_calls = 0
    callback_closed = False
    trace = []
    observed_metadata = []
    native_packets = {}
    buffers = []

    # Separate direct-file reading supplies metadata and timestamp expectations.
    reference = []
    reference_ds = DataSource(
        files=str(xtc_dir / "xpptut15-r0014-s000-c000.xtc2"),
        skip_calib_load="all", monitor=False,
    )
    for run in reference_ds.runs():
        for step in run.steps():
            for evt in step.events():
                reference.append(metadata(run, step, evt))

    def callback(run):
        nonlocal callback_calls, callback_closed
        callback_calls += 1
        try:
            if behavior == "events":
                for ievt, evt in enumerate(run.events()):
                    trace.append([0, ievt, int(evt.timestamp)])
                    observed_metadata.append(metadata(run, None, evt))
                    if 2 <= ievt < 5:
                        evt.set_destination(2)
                        yield evt
            else:
                for istep, step in enumerate(run.steps()):
                    for ievt, evt in enumerate(step.events()):
                        trace.append([istep, ievt, int(evt.timestamp)])
                        observed_metadata.append(metadata(run, step, evt))
                        if behavior == "error" and len(trace) == 5:
                            raise ValueError("callback test error")
                        if behavior != "reject" and (stepnum is None or istep == stepnum) and 2 <= ievt < 5:
                            evt.set_destination(2)
                            yield evt
                            if behavior == "return":
                                return
        finally:
            callback_closed = True

    class RecordingBuilder:
        def __init__(self, native):
            self.native = native

        def has_more(self):
            return self.native.has_more()

        def events(self):
            for dgrams, proxy in self.native.events():
                native_packets[proxy.timestamp] = proxy.as_bytearray()
                yield dgrams, proxy

    def poison():
        for buf in buffers:
            buf[:] = b"\xa5" * len(buf)

    with patch.object(DataSourceBase, "_get_file_info_from_db", return_value={}):
        ds = DataSource(exp="xpptut15", run=14, dir=str(xtc_dir),
                        skip_calib_load="all", monitor=False, batch_size=batch_size)
        run = next(ds.runs())  # Consume Configure/BeginRun, no production event loop.
        manager = run.smdr_man
        configs = manager.configs
        parms = ds.dsparms

        def builders():
            nonlocal buffers, source_chunks, source_closed
            try:
                for _ in manager.chunks():
                    poison()
                    buffers = [bytearray(manager.smdr.show(i)) for i in range(manager.n_files)]
                    source_chunks += 1
                    yield RecordingBuilder(EventBuilder(
                        buffers, configs, filter_timestamps=parms.timestamps,
                        intg_stream_id=parms.intg_stream_id, batch_size=batch_size,
                        use_proxy_events=True,
                    ))
            finally:
                poison()
                source_closed = True

        class ClosingSource:
            def __init__(self):
                self.iterator = builders()

            def __iter__(self):
                return self

            def __next__(self):
                return next(self.iterator)

            def close(self):
                nonlocal source_close_calls
                source_close_calls += 1
                self.iterator.close()

        iterator = CallbackIterator(ClosingSource(), configs, parms, callback)
        selected = []
        caught = None
        if behavior in ("close", "before_start"):
            if behavior == "close":
                selected.append(next(iterator))
            chunks_before_close = source_chunks
            iterator.close()
            iterator.close()  # Closing is idempotent and must not refill input.
            assert source_chunks == chunks_before_close
        else:
            try:
                selected.extend(iterator)
            except ValueError as exc:
                caught = str(exc)

        assert list(iterator) == []  # Exhaustion/return/error never restarts it.
        selected_timestamps = [int(evt.timestamp) for evt in selected]
        assert all(evt._proxy_evt.destination == 2 for evt in selected)
        records = [evt._proxy_evt for evt in selected] + iterator.run.proxy_events
        # All chunk buffers have been overwritten by now. Compare full packets,
        # including footers and payloads, against the native pre-refill encoding.
        assert all(record.as_bytearray() == native_packets[record.timestamp] for record in records)
        assert observed_metadata == reference[:len(observed_metadata)]
        result = dict(
            selected=selected_timestamps, trace=trace, callback_calls=callback_calls,
            source_chunks=source_chunks, chunks_read=iterator.chunks_read,
            source_closed=source_closed, callback_closed=callback_closed,
            caught=caught, transition_count=len(iterator.run.proxy_events),
            source_close_calls=source_close_calls,
        )
        iterator.close()
        assert iterator.run.proxy_events == []
        assert source_close_calls == 1
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xtc_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stepnum", type=int)
    parser.add_argument("--behavior", choices=["steps", "events", "reject", "return", "close", "before_start", "error"], default="steps")
    args = parser.parse_args()
    # A new iterator/run must start fresh even when the process is reused.
    results = [exercise(args.xtc_dir, args.batch_size, args.stepnum, args.behavior) for _ in range(2)]
    assert results[0] == results[1]
    print("CALLBACK_ITERATOR_RESULT=" + json.dumps(results[0]))


if __name__ == "__main__":
    main()
