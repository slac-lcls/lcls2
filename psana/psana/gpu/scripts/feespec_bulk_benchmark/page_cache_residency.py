#!/usr/bin/env python3
"""Report Linux page-cache residency for benchmark input files."""

import argparse
import ctypes
import json
import mmap
import os

import numpy as np


def file_residency(path):
    page_size = os.sysconf("SC_PAGE_SIZE")
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        n_pages = (size + page_size - 1) // page_size
        mapping = mmap.mmap(
            fd,
            size,
            flags=mmap.MAP_PRIVATE,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
    finally:
        os.close(fd)

    vector = (ctypes.c_ubyte * n_pages)()
    anchor = ctypes.c_char.from_buffer(mapping)
    address = ctypes.addressof(anchor)
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.mincore(
        ctypes.c_void_p(address), ctypes.c_size_t(size), ctypes.byref(vector)
    )
    if result != 0:
        error = ctypes.get_errno()
        del anchor
        mapping.close()
        raise OSError(error, os.strerror(error), path)

    resident_pages = int(
        np.bitwise_and(np.ctypeslib.as_array(vector), 1).sum(dtype=np.uint64)
    )
    del anchor
    mapping.close()
    return {
        "path": path,
        "bytes": size,
        "pages": n_pages,
        "resident_pages": resident_pages,
        "resident_fraction": resident_pages / n_pages,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--minimum",
        type=float,
        default=None,
        help="fail if aggregate resident page fraction is below this value",
    )
    args = parser.parse_args()

    files = [file_residency(path) for path in args.paths]
    total_pages = sum(item["pages"] for item in files)
    resident_pages = sum(item["resident_pages"] for item in files)
    record = {
        "tag": args.tag,
        "files": files,
        "pages": total_pages,
        "resident_pages": resident_pages,
        "resident_fraction": resident_pages / total_pages,
    }
    print("CACHE_RESIDENCY " + json.dumps(record, sort_keys=True), flush=True)
    if args.minimum is not None and record["resident_fraction"] < args.minimum:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
