"""Subprocess driver for test_callback_chunk_state.py (serial or one MPI EB)."""

import argparse
import json
from pathlib import Path
import threading
from unittest.mock import patch

from psana import DataSource
from psana.psexp.ds_base import DataSourceBase
from psana.psexp.tools import mode
from psana.psexp.callback_stream import StreamingCallbackBatches


def metadata(run, step, evt):
    return [int(evt.timestamp), run.Detector("motor2")(step),
            run.Detector("HX2:DVD:GCC:01:PMON,hello1")(evt)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xtc_dir", type=Path)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stepnum", type=int)
    parser.add_argument("--evskip", type=int, default=2)
    parser.add_argument("--nrecs", type=int, default=5)
    parser.add_argument("--behavior", choices=["select", "reject", "return"], default="select")
    parser.add_argument("--destinations", action="store_true")
    parser.add_argument("--small-limits", action="store_true")
    parser.add_argument("--twice", action="store_true")
    parser.add_argument("--max-events", type=int, default=0)
    args = parser.parse_args()

    comm = None
    if mode == "mpi":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        assert comm.size == 4, "Use one SMD0, one EB, and two BD ranks"

    if args.reference:
        assert comm is None, "The direct-file oracle must run serially"
        # Stream 0 contains every event and all scan transitions. Direct-file
        # reading bypasses SmdReaderManager, EventBuilder and the callback.
        ds = DataSource(
            files=str(args.xtc_dir / "xpptut15-r0014-s000-c000.xtc2"),
            skip_calib_load="all",
            monitor=False,
        )
        steps = []
        reference_metadata = []
        for run in ds.runs():
            for step in run.steps():
                timestamps = []
                for evt in step.events():
                    timestamps.append(int(evt.timestamp))
                    reference_metadata.append(metadata(run, step, evt))
                steps.append(timestamps)
        result = {"steps": steps, "metadata": reference_metadata}
    else:
        trace = []
        invocations = 0
        streams = []
        output_metadata = []
        callback_metadata = []
        step_counts = []
        destinations = []
        received = []
        callback_thread = threading.get_ident()
        original_init = StreamingCallbackBatches.__init__

        def tracked_init(self, *pos, **kw):
            if args.small_limits:
                kw.update(max_records=3, max_bytes=512)
            original_init(self, *pos, **kw)
            streams.append(self)

        def callback(run):
            nonlocal invocations
            invocations += 1
            assert threading.get_ident() == callback_thread
            # These counters intentionally live inside the generator, just as
            # in the Jungfrau callbacks. Do not carry them across invocations.
            for istep, step in enumerate(run.steps()):
                for ievt, evt in enumerate(step.events()):
                    trace.append([invocations, istep, ievt, int(evt.timestamp)])
                    callback_metadata.append(metadata(run, step, evt))
                    if args.behavior != "reject" and (args.stepnum is None or istep == args.stepnum) and (
                        args.evskip <= ievt < args.nrecs
                    ):
                        if args.destinations:
                            dest = ievt % 3  # Mix default routing with both explicit BDs.
                            evt.set_destination(dest)
                            destinations.append([int(evt.timestamp), dest])
                        yield evt
                        if args.behavior == "return":
                            return

        # Synthetic files have no logbook entry. Stub only external discovery;
        # all SMD chunking, callback iteration and bigdata reading remain real.
        with patch.object(DataSourceBase, "_get_file_info_from_db", return_value={}), \
                patch.object(StreamingCallbackBatches, "__init__", tracked_init):
            ds = DataSource(
                exp="xpptut15",
                run=[14, 14] if args.twice else 14,
                dir=str(args.xtc_dir),
                smd_callback=callback,
                batch_size=args.batch_size,
                skip_calib_load="all",
                monitor=False,
                max_events=args.max_events,
            )
            selected = []
            for run in ds.runs():
                nsteps = 0
                for step in run.steps():
                    nsteps += 1
                    for evt in step.events():
                        selected.append(int(evt.timestamp))
                        output_metadata.append(metadata(run, step, evt))
                        received.append([int(evt.timestamp), comm.rank - 1 if comm else 0])
                step_counts.append(nsteps)

        result = dict(selected=selected, trace=trace, invocations=invocations,
                      chunks=sum(stream.iterator.chunks_read for stream in streams),
                      metadata=output_metadata, callback_metadata=callback_metadata,
                      step_counts=step_counts, destinations=destinations, received=received)
        if comm is not None:
            results = comm.gather(result, root=0)
            if comm.rank != 0:
                return
            result = {
                "selected": sorted(ts for item in results for ts in item["selected"]),
                "trace": [row for item in results for row in item["trace"]],
                "invocations": sum(item["invocations"] for item in results),
                "chunks": sum(item["chunks"] for item in results),
                "metadata": sorted(row for item in results for row in item["metadata"]),
                "callback_metadata": [row for item in results for row in item["callback_metadata"]],
                "step_counts": [count for item in results for count in item["step_counts"] if count],
                "destinations": [row for item in results for row in item["destinations"]],
                "received": [row for item in results for row in item["received"]],
            }

    print("CALLBACK_CHUNK_RESULT=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
