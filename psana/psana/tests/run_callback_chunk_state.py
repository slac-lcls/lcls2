"""Subprocess driver for test_callback_chunk_state.py (serial or one MPI EB)."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from psana import DataSource
from psana.psexp.ds_base import DataSourceBase
from psana.psexp.tools import mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xtc_dir", type=Path)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stepnum", type=int)
    parser.add_argument("--evskip", type=int, default=2)
    parser.add_argument("--nrecs", type=int, default=5)
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
        for run in ds.runs():
            for step in run.steps():
                steps.append([int(evt.timestamp) for evt in step.events()])
        result = {"steps": steps}
    else:
        trace = []
        invocations = 0

        def callback(run):
            nonlocal invocations
            invocations += 1
            # These counters intentionally live inside the generator, just as
            # in the Jungfrau callbacks. Do not carry them across invocations.
            for istep, step in enumerate(run.steps()):
                for ievt, evt in enumerate(step.events()):
                    trace.append([invocations, istep, ievt, int(evt.timestamp)])
                    if (args.stepnum is None or istep == args.stepnum) and (
                        args.evskip <= ievt < args.nrecs
                    ):
                        yield evt

        # Synthetic files have no logbook entry. Stub only external discovery;
        # all SMD chunking, callback iteration and bigdata reading remain real.
        with patch.object(DataSourceBase, "_get_file_info_from_db", return_value={}):
            ds = DataSource(
                exp="xpptut15",
                run=14,
                dir=str(args.xtc_dir),
                smd_callback=callback,
                batch_size=args.batch_size,
                skip_calib_load="all",
                monitor=False,
            )
            selected = [int(evt.timestamp) for run in ds.runs() for evt in run.events()]

        result = {"selected": selected, "trace": trace, "invocations": invocations}
        if comm is not None:
            results = comm.gather(result, root=0)
            if comm.rank != 0:
                return
            result = {
                "selected": sorted(ts for item in results for ts in item["selected"]),
                "trace": [row for item in results for row in item["trace"]],
                "invocations": sum(item["invocations"] for item in results),
            }

    print("CALLBACK_CHUNK_RESULT=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
