#!/usr/bin/env python

"""Extract a sparse subset of events from xtc2 and smd.xtc2 stream files.

The script uses ``DataSource(..., detectors=[...])`` to resolve the stream
files for a given run, then copies raw datagrams while preserving every
non-event transition exactly as-is. Only event datagrams are filtered.

Example
-------
python psana/psana/app/extract_subset_xtc2.py \
    --exp mfx100848724 \
    --run 54 \
    --dir /sdf/data/lcls/ds/mfx/mfx100848724/xtc \
    --detectors jungfrau \
    --events 5296,14070,17142,19632,23232,23616 \
    --output-dir ./mfx100848724-r0054-subset
"""

from __future__ import print_function

import argparse
import os
import shutil
import struct
import subprocess
import time


HEADER_WORDS = 6
HEADER_BYTES = HEADER_WORDS * 4
XTC_HEADER_BYTES = 12

TRANSITION_CONFIGURE = 2
TRANSITION_L1ACCEPT_ENDOFBATCH = 11
TRANSITION_L1ACCEPT = 12
EVENT_TRANSITIONS = (TRANSITION_L1ACCEPT_ENDOFBATCH, TRANSITION_L1ACCEPT)


def parse_event_indices(spec):
    """Parse a zero-based comma-separated event list with inclusive ranges."""
    indices = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start < 0 or end < 0:
                raise ValueError("Event indices must be non-negative")
            if end < start:
                raise ValueError("Invalid event range: %s" % token)
            indices.update(range(start, end + 1))
        else:
            index = int(token)
            if index < 0:
                raise ValueError("Event indices must be non-negative")
            indices.add(index)
    if not indices:
        raise ValueError("No event indices were parsed from %r" % spec)
    return indices


def parse_detector_names(specs):
    """Parse detector names from space-separated args and/or comma-separated tokens."""
    detectors = []
    for spec in specs or []:
        for detector in spec.split(","):
            detector = detector.strip()
            if detector:
                detectors.append(detector)
    return detectors


def read_datagram(infile):
    """Read one raw xtc2 datagram and return ``(raw_bytes, service, timestamp)``."""
    header = infile.read(HEADER_BYTES)
    if not header:
        return None
    if len(header) != HEADER_BYTES:
        raise IOError("Truncated xtc2 header")

    words = struct.unpack("<%dI" % HEADER_WORDS, header)
    time_low = words[0]
    time_high = words[1]
    env_word = words[2]
    extent = words[5]
    payload_size = extent - XTC_HEADER_BYTES
    if payload_size < 0:
        raise ValueError("Invalid xtc2 extent %d" % extent)

    payload = infile.read(payload_size)
    if len(payload) != payload_size:
        raise IOError("Truncated xtc2 payload")

    service = (env_word >> 24) & 0xF
    timestamp = (time_high << 32) | time_low
    return header + payload, service, timestamp


def copy_filtered_xtc_file(
    input_path,
    output_path,
    event_indices=None,
    num_events=None,
    selected_timestamps=None,
):
    """Copy one xtc2 stream, preserving all transitions and filtering events."""
    selection_modes = [
        event_indices is not None,
        num_events is not None,
        selected_timestamps is not None,
    ]
    if sum(selection_modes) != 1:
        raise ValueError(
            "Exactly one of event_indices, num_events, or selected_timestamps must be provided"
        )

    total_events = 0
    written_events = 0
    saw_configure = False

    with open(input_path, "rb") as infile:
        with open(output_path, "wb") as outfile:
            while True:
                datagram = read_datagram(infile)
                if datagram is None:
                    break

                raw_bytes, service, timestamp = datagram
                if not saw_configure:
                    saw_configure = True
                    if service != TRANSITION_CONFIGURE:
                        raise ValueError(
                            "Expected first datagram in %s to be Configure, got service %d"
                            % (input_path, service)
                        )

                if service in EVENT_TRANSITIONS:
                    keep_event = False
                    if selected_timestamps is not None:
                        keep_event = timestamp in selected_timestamps
                    elif event_indices is not None:
                        keep_event = total_events in event_indices
                    else:
                        keep_event = total_events < num_events

                    if keep_event:
                        outfile.write(raw_bytes)
                        written_events += 1
                    total_events += 1
                else:
                    outfile.write(raw_bytes)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "total_events": total_events,
        "written_events": written_events,
    }


def extract_selected_streams(
    xtc_files,
    smd_files,
    output_dir,
    event_indices=None,
    num_events=None,
    selected_timestamps=None,
):
    """Write filtered bigdata and smalldata streams into ``output_dir``."""
    if not xtc_files:
        raise ValueError("No bigdata xtc2 files were selected")
    if not smd_files:
        raise ValueError("No smalldata xtc2 files were selected")

    output_dir = os.path.abspath(output_dir)
    smd_output_dir = os.path.join(output_dir, "smalldata")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(smd_output_dir):
        os.makedirs(smd_output_dir)

    results = {"xtc": [], "smd": []}
    smdwriter_bin = shutil.which("smdwriter")
    if smdwriter_bin is None:
        raise RuntimeError("smdwriter was not found on PATH")

    for input_path in xtc_files:
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        xtc_result = copy_filtered_xtc_file(
            input_path,
            output_path,
            event_indices=event_indices,
            num_events=num_events,
            selected_timestamps=selected_timestamps,
        )
        results["xtc"].append(xtc_result)

        smd_output_path = os.path.join(
            smd_output_dir,
            os.path.splitext(os.path.basename(input_path))[0] + ".smd.xtc2",
        )
        subprocess.run(
            [smdwriter_bin, "-f", output_path, "-o", smd_output_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        results["smd"].append(
            {
                "input_path": output_path,
                "output_path": smd_output_path,
                "total_events": xtc_result["total_events"],
                "written_events": xtc_result["written_events"],
            }
        )

    return results


def _make_index_selection_callback(event_indices=None, num_events=None):
    """Build an SMD callback that yields only the requested L1 events."""
    if event_indices is None and num_events is None:
        raise ValueError("Either event_indices or num_events must be provided")
    if event_indices is not None and num_events is not None:
        raise ValueError("event_indices and num_events are mutually exclusive")

    selected = set(event_indices) if event_indices is not None else None
    max_index = max(selected) if selected else None
    found_indices = set()
    next_index = [0]
    selected_pairs = []

    def smd_callback(run):
        for evt in run.events():
            event_index = next_index[0]
            next_index[0] += 1
            if selected is not None:
                if event_index in selected:
                    found_indices.add(event_index)
                    selected_pairs.append((event_index, evt.timestamp))
                    yield evt
                if max_index is not None and event_index >= max_index:
                    return
            else:
                if event_index < num_events:
                    selected_pairs.append((event_index, evt.timestamp))
                    yield evt
                else:
                    return

    smd_callback.found_indices = found_indices
    smd_callback.selected_pairs = selected_pairs
    return smd_callback


def collect_selected_timestamps(exp, run, xtc_dir, detectors=None, event_indices=None, num_events=None):
    """Collect selected event timestamps using an SMD-side callback."""
    if event_indices is None and num_events is None:
        raise ValueError("Either event_indices or num_events must be provided")
    if event_indices is not None and num_events is not None:
        raise ValueError("event_indices and num_events are mutually exclusive")

    if "PS_PARALLEL" not in os.environ:
        os.environ["PS_PARALLEL"] = "none"

    from psana import DataSource

    kwargs = {"exp": exp, "run": run}
    if xtc_dir:
        kwargs["dir"] = xtc_dir
    if detectors:
        kwargs["detectors"] = detectors
    if event_indices:
        kwargs["max_events"] = max(event_indices) + 1
    elif num_events is not None:
        kwargs["max_events"] = num_events
    callback = _make_index_selection_callback(
        event_indices=event_indices,
        num_events=num_events,
    )
    kwargs["smd_callback"] = callback
    kwargs["skip_calib_load"] = "all"
    kwargs["batch_size"] = 1

    ds = DataSource(**kwargs)
    xtc_files = list(getattr(ds, "xtc_files", []) or [])
    smd_files = list(getattr(ds, "smd_files", []) or [])

    run_iter = ds.runs()
    run_obj = next(run_iter, None)
    if run_obj is None:
        raise ValueError("Run %d was not found for experiment %s" % (run, exp))

    for evt in run_obj.events():
        pass

    selected_pairs = list(callback.selected_pairs)
    timestamps = [timestamp for _event_index, timestamp in selected_pairs]

    if event_indices is not None:
        missing = sorted(event_indices.difference(callback.found_indices))
        if missing:
            raise ValueError(
                "Requested event indices are outside the selected DataSource event stream: %s"
                % missing
            )
    elif num_events is not None and len(timestamps) != num_events:
        raise ValueError(
            "Requested %d events but only found %d in the selected DataSource event stream"
            % (num_events, len(timestamps))
        )

    return selected_pairs, xtc_files, smd_files


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Extract sparse regression subsets from xtc2 and smd.xtc2 files."
    )
    parser.add_argument("--exp", required=True, help="Experiment name, e.g. mfx100848724")
    parser.add_argument("--run", required=True, type=int, help="Run number")
    parser.add_argument("--dir", help="Optional run xtc directory")
    parser.add_argument(
        "--detectors",
        nargs="*",
        default=[],
        help="Optional detector selection used to prune stream files; accepts space- or comma-separated names",
    )
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        "--events",
        help="Zero-based event indices or ranges, e.g. 0,10,20-25",
    )
    selection_group.add_argument(
        "--num-events",
        type=int,
        help="Copy the first N events from each selected stream",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. Bigdata files go here; smalldata files go in output_dir/smalldata",
    )
    return parser


def format_summary(kind, results):
    lines = []
    for result in results:
        lines.append(
            "%s %s: wrote %d / %d events"
            % (
                kind,
                os.path.basename(result["output_path"]),
                result["written_events"],
                result["total_events"],
            )
        )
    return lines


def format_index_timestamp_pairs(selected_pairs):
    lines = []
    for event_index, timestamp in selected_pairs:
        lines.append("event_index %d -> timestamp %d" % (event_index, timestamp))
    return lines


def main(argv=None):
    start_time = time.monotonic()
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.num_events is not None and args.num_events < 0:
        parser.error("--num-events must be non-negative")

    event_indices = parse_event_indices(args.events) if args.events else None
    detectors = parse_detector_names(args.detectors)

    selected_pairs, xtc_files, smd_files = collect_selected_timestamps(
        exp=args.exp,
        run=args.run,
        xtc_dir=args.dir,
        detectors=detectors,
        event_indices=event_indices,
        num_events=args.num_events,
    )

    if not xtc_files:
        parser.error("No bigdata xtc2 files matched the requested run/detector selection")
    if not smd_files:
        parser.error("No smalldata xtc2 files matched the requested run/detector selection")

    results = extract_selected_streams(
        xtc_files=xtc_files,
        smd_files=smd_files,
        output_dir=args.output_dir,
        selected_timestamps=set(
            timestamp for _event_index, timestamp in selected_pairs
        ),
    )

    print("Selected event index -> timestamp mapping:")
    for line in format_index_timestamp_pairs(selected_pairs):
        print(line)
    print(
        "Wrote %d bigdata and %d smalldata streams to %s"
        % (len(results["xtc"]), len(results["smd"]), os.path.abspath(args.output_dir))
    )
    for line in format_summary("xtc", results["xtc"]):
        print(line)
    for line in format_summary("smd", results["smd"]):
        print(line)
    print("Elapsed %.3f s" % (time.monotonic() - start_time))


if __name__ == "__main__":
    main()
