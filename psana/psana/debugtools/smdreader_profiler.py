#!/usr/bin/env python
"""
Utility to profile the SmdReader by sweeping over different numbers of
smalldata (.smd.xtc2) files. The tool prints per-run timing information such
as the internal search time reported by the reader, wall-clock elapsed time,
and event processing rate.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Iterable, Sequence

import numpy as np
from mpi4py import MPI

from psana.psexp.ds_base import DsParms
from psana.psexp.smdreader_manager import SmdReaderManager

COMM = MPI.COMM_WORLD
HOST = MPI.Get_processor_name()


def _create_default_dsparms() -> DsParms:
    # NOTE: Update this helper if DsParms adds/removes required fields.
    return DsParms(
        batch_size=1,
        max_events=0,
        max_retries=0,
        live=False,
        timestamps=np.empty(0, dtype=np.uint64),
        intg_det="",
        intg_delta_t=0,
        use_calib_cache=False,
        cached_detectors=[],
        fetch_calib_cache_max_retries=60,
        skip_calib_load=[],
        dbsuffix="",
    )


def _positive_int(value: str) -> int:
    intval = int(value)
    if intval <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer.")
    return intval


def _derive_counts(max_files: int, mode: str, overrides: Sequence[int] | None) -> list[int]:
    if overrides:
        return sorted(set(overrides))

    if mode == "linear":
        return list(range(1, max_files + 1))

    counts = []
    current = 1
    while current < max_files:
        counts.append(current)
        current *= 2
    counts.append(max_files)
    return counts


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile SmdReader throughput across multiple numbers of smd files.\n\n"
            "Example:\n"
            "  mpirun -n 1 python -m psana.debugtools.smdreader_profiler "
            "--files-glob '/path/to/*.smd.xtc2' --max-files 60 --count-mode powers\n"
            "Consider allocating a dedicated compute node before running."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--files-glob",
        required=True,
        help="Glob describing smd xtc2 files (e.g. '/path/run/*.smd.xtc2').",
    )
    parser.add_argument(
        "--counts",
        type=_positive_int,
        nargs="+",
        help="Exact file counts to run. Overrides --max-files/--count-mode when provided.",
    )
    parser.add_argument(
        "--single-count",
        type=_positive_int,
        help=(
            "Profile only this number of smd files. When set, overrides --counts, "
            "--max-files, and --count-mode."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=_positive_int,
        default=32,
        help="Upper bound used when deriving file counts (default: 32).",
    )
    parser.add_argument(
        "--count-mode",
        choices=("powers", "linear"),
        default="powers",
        help=(
            "When --counts is not supplied, choose between powers-of-two or linear sweep "
            "up to --max-files (default: powers)."
        ),
    )
    parser.add_argument(
        "--quiet-summary",
        action="store_true",
        help="Suppress the summary table printed after all runs.",
    )
    return parser.parse_args(argv)


def _open_fd(path: str) -> int:
    """Open a file descriptor with O_DIRECT when available, falling back to O_RDONLY."""
    flags = os.O_RDONLY
    o_direct = getattr(os, "O_DIRECT", 0)
    if o_direct:
        try:
            return os.open(path, flags | o_direct)
        except OSError:
            pass
    return os.open(path, flags)


def _profile_once(filepaths: Sequence[str], dsparms: DsParms) -> dict[str, float]:
    """Return timing information for a single file-count run."""
    dsparms.update_smd_state(list(filepaths), [True] * len(filepaths))
    fds = np.array([_open_fd(path) for path in filepaths], dtype=np.int32)
    start = time.perf_counter()
    try:
        smdr_man = SmdReaderManager(fds, dsparms)
        smdr_man.get_next_dgrams()
        smdr_man.get_next_dgrams()
        for _ in smdr_man.chunks():
            # Force SmdReader to walk all step buffers for each file.
            _ = [smdr_man.smdr.show(i, step_buf=True) for i in range(len(filepaths))]
            if not smdr_man.got_events or smdr_man.smdr.found_endrun():
                break
        elapsed = time.perf_counter() - start
        processed_events = smdr_man.processed_events
        if dsparms.max_events > 0 and processed_events > dsparms.max_events:
            processed_events = dsparms.max_events
        rate = processed_events / (elapsed * 1e6) if elapsed else 0.0
        search_time = getattr(smdr_man.smdr, "total_time", float("nan"))
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
    return {
        "files": len(filepaths),
        "search_time": search_time,
        "elapsed": elapsed,
        "events": processed_events,
        "rate": rate,
    }


def _print_run(stats: dict[str, float]) -> None:
    thread_cnt = int(os.environ.get("PS_SMD0_NUM_THREADS", 16))
    print(f"total search time: {stats['search_time']}")
    print(
        "Host: {host} #Files: {files} #Threads: {threads} #Events: {events} "
        "Elapsed Time (s): {elapsed:.2f} Rate (MHz): {rate:.2f}".format(
            host=HOST,
            files=stats["files"],
            threads=thread_cnt,
            events=int(stats["events"]),
            elapsed=stats["elapsed"],
            rate=stats["rate"],
        )
    )


def _print_summary(results: Iterable[dict[str, float]]) -> None:
    header = "Files  Search(s)  Elapsed(s)  Rate(MHz)  Events"
    print("\nSummary:")
    print(header)
    for stats in results:
        print(
            f"{stats['files']:>5}  "
            f"{stats['search_time']:>9.2f}  "
            f"{stats['elapsed']:>10.2f}  "
            f"{stats['rate']:>8.2f}  "
            f"{int(stats['events']):>6}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ.setdefault("PS_SMD_MAX_RETRIES", "0")

    globbed_files = sorted(glob.glob(args.files_glob))
    if not globbed_files:
        print(f"No files matched glob: {args.files_glob}", file=sys.stderr)
        return 1

    if args.single_count:
        counts = [args.single_count]
    else:
        counts = _derive_counts(args.max_files, args.count_mode, args.counts)
    if max(counts) > len(globbed_files):
        print(
            f"Requested {max(counts)} files but only {len(globbed_files)} exist for glob {args.files_glob}.",
            file=sys.stderr,
        )
        return 1

    results: list[dict[str, float]] = []
    for count in counts:
        subset = globbed_files[:count]
        dsparms = _create_default_dsparms()
        stats = _profile_once(subset, dsparms)
        results.append(stats)
        _print_run(stats)

    if not args.quiet_summary:
        _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
