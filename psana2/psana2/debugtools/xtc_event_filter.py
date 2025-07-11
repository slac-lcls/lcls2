"""
xtc_event_filter.py

Utility to copy and trim an xtc2 file by selecting a subset of events.

Can be used as a CLI tool:
    python -m psana.psana.debugtools.xtc_event_filter input.xtc2 -n 10 --select-events 2-5,8

Or imported and reused in test scripts.
"""

import argparse
import os

from psana2.dgram import Dgram
from psana2.psexp import TransitionId


def open_file(fname):
    """Open a file and return file descriptor and size."""
    fd = os.open(fname, os.O_RDONLY)
    f_size = os.path.getsize(fname)
    return fd, f_size

def get_config(fd_in):
    """Read the first datagram (e.g., Configure) and return offset and object."""
    config = Dgram(file_descriptor=fd_in)
    offset = memoryview(config).nbytes
    return offset, config

def build_output_filename(input_path):
    """Generate output filename by appending _modified before extension."""
    base, ext = os.path.splitext(input_path)
    return f"{base}_modified{ext}"

def parse_event_selection(event_str):
    """Parse '2-5,8,12-14' into a set of selected event integers."""
    selected = set()
    for part in event_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            selected.update(range(int(start), int(end)+1))
        else:
            selected.add(int(part))
    return selected

def run_xtc_modifier(input_file, output_file=None, num_events=None, select_events=None):
    """
    Trim an xtc2 file to a subset of L1Accept events, keeping essential transitions for valid structure.

    Args:
        input_file (str): Path to the input xtc2 file.
        output_file (str): Output file path (optional).
        num_events (int): Max number of L1Accept events to write.
        select_events (set): Specific L1Accept event numbers to keep.
    """
    output_file = output_file or build_output_filename(input_file)
    selected_events = parse_event_selection(select_events) if select_events else None

    xtc_fd, xtc_f_size = open_file(input_file)
    with open(output_file, "wb", buffering=0) as xtc_f_out:
        xtc_offset, xtc_config = get_config(xtc_fd)
        xtc_f_out.write(xtc_config)
        print(f"Wrote Configure datagram ({memoryview(xtc_config).nbytes} bytes)")

        cn_events = 1
        write_events = True
        final_transitions = []
        seen_transitions = set()

        while xtc_offset < xtc_f_size:
            d = Dgram(config=xtc_config)
            xtc_offset += memoryview(d).nbytes
            service = d.service()

            # Skip SlowUpdate completely
            if service == TransitionId.SlowUpdate:
                continue

            # Only write first of each of these transitions
            if service in {TransitionId.BeginRun, TransitionId.BeginStep, TransitionId.Enable}:
                if service in seen_transitions:
                    continue
                seen_transitions.add(service)
                xtc_f_out.write(d)
                print(f"Wrote first {TransitionId.name(service)} transition: {memoryview(d).nbytes} bytes")
                continue

            # Write L1Accepts based on selection criteria
            if service == TransitionId.L1Accept:
                if write_events:
                    if num_events and cn_events > num_events:
                        write_events = False
                        print(f"Reached L1Accept limit of {num_events}. Skipping further events.")
                        cn_events += 1
                        continue
                    if selected_events and cn_events not in selected_events:
                        cn_events += 1
                        continue
                    xtc_f_out.write(d)
                    print(f"Wrote L1Accept event {cn_events}: {memoryview(d).nbytes} bytes")
                    cn_events += 1
                else:
                    cn_events += 1
                continue

            # Use a dict to buffer specific transitions in final order
            if service in {TransitionId.Disable, TransitionId.EndStep, TransitionId.EndRun}:
                if service not in seen_transitions:
                    seen_transitions.add(service)
                    final_transitions.append((service, d))  # Will be sorted later
                continue

            # Write any other (rare) transition types
            xtc_f_out.write(d)
            print(f"Wrote {TransitionId.name(service)} transition: {memoryview(d).nbytes} bytes")

        # Write Disable, EndStep, EndRun — in that order if present
        transition_priority = [TransitionId.Disable, TransitionId.EndStep, TransitionId.EndRun]
        final_map = dict(final_transitions)

        for tid in transition_priority:
            if tid in final_map:
                d = final_map[tid]
                xtc_f_out.write(d)
                print(f"Wrote final transition {TransitionId.name(d.service())}: {memoryview(d).nbytes} bytes")

    os.close(xtc_fd)
    written = min(cn_events - 1, num_events) if num_events else cn_events - 1
    print(f"Done. Wrote {written} L1Accept event(s) to {output_file}")

def main():
    """Entry point for CLI usage."""
    parser = argparse.ArgumentParser(description="Modify xtc2 file by selecting events.")
    parser.add_argument("input", help="Input xtc2 filename")
    parser.add_argument("-o", "--output", help="Output xtc2 filename (default: <input>_modified.xtc2)")
    parser.add_argument("-n", "--num-events", type=int, help="Number of events to write")
    parser.add_argument("--select-events", help="Comma-separated event ranges to keep, e.g., 2-5,8,12-14")

    args = parser.parse_args()
    run_xtc_modifier(args.input, args.output, args.num_events, args.select_events)

if __name__ == "__main__":
    main()
