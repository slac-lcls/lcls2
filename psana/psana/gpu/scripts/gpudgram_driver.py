#!/usr/bin/env python3
"""Isolated KvikIO -> GPUDgram correctness driver.

The CPU reads only the leading Configure dgram and compiles its Names schema.
KvikIO reads a contiguous byte range after Configure into GPU memory.  GPU
kernels discover dgram boundaries, walk nested XTC Parents, and locate fields.
CPU event parsing is used afterward only as a verification oracle.
"""

import argparse
import os

import numpy as np

from psana import dgram
from psana.gpu.gpudgram import GPUDgramBatch, GpuNamesSchema
from psana.gpu.gpudgram.parser import (
    DGRAM_OFFSET,
    DGRAM_SIZE,
    STATUS_NAMES,
)


SERVICE_NAMES = {
    0: "ClearReadout",
    1: "Reset",
    2: "Configure",
    3: "Unconfigure",
    4: "BeginRun",
    5: "EndRun",
    6: "BeginStep",
    7: "EndStep",
    8: "Enable",
    9: "Disable",
    10: "SlowUpdate",
    12: "L1Accept",
}


def _int_arg(value):
    return int(value, 0)


def _read_config(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        config = dgram.Dgram(file_descriptor=fd)
    finally:
        os.close(fd)
    if int(config.service()) != 2:
        raise RuntimeError(
            f"first dgram in {path!r} is service {config.service()}, not Configure"
        )
    return config


def _kvikio_read(path, file_offset, nbytes):
    import cupy as cp
    import kvikio

    data_gpu = cp.empty(nbytes, dtype=cp.uint8)
    cu_file = kvikio.CuFile(path, "r")
    try:
        nread = int(
            cu_file.pread(
                data_gpu,
                size=nbytes,
                file_offset=file_offset,
            ).get()
        )
    finally:
        cu_file.close()
    if nread != nbytes:
        raise RuntimeError(f"KvikIO requested {nbytes} bytes but read {nread}")
    return data_gpu


def _verify_dgram(
    path,
    config,
    config_size,
    batch,
    index,
    gpu_alg,
    field_name,
    verify_max_bytes,
):
    import cupy as cp

    info = batch.dgram_info[index]
    file_offset = config_size + int(info[DGRAM_OFFSET])
    dgram_size = int(info[DGRAM_SIZE])
    fd = os.open(path, os.O_RDONLY)
    try:
        cpu_dgram = dgram.Dgram(
            file_descriptor=fd,
            config=config,
            offset=file_offset,
            size=dgram_size,
        )
        cpu_descs = cpu_dgram.raw_descriptors(
            config=config,
            det_name=gpu_alg.det_name,
            alg_name=gpu_alg.alg_name,
            field_name=field_name,
        )
        cpu_by_field = {
            desc["field_name"]: desc
            for desc in cpu_descs
            if int(desc["segment"]) == gpu_alg.segment
        }

        selected_fields = (
            [field_name] if field_name is not None else list(gpu_alg.fields)
        )
        for name in selected_fields:
            gpu_field = gpu_alg[name]
            cpu_desc = cpu_by_field.get(name)
            if cpu_desc is None:
                raise AssertionError(f"CPU parser did not find field {name!r}")
            expected_gpu_offset = (
                int(info[DGRAM_OFFSET]) + int(cpu_desc["field_rel_offset"])
            )
            assert gpu_field.device_offset == expected_gpu_offset
            assert gpu_field.nbytes == int(cpu_desc["field_nbytes"])
            assert gpu_field.shape == tuple(cpu_desc["shape"])
            assert gpu_field.type == int(cpu_desc["type"])

            data_result = "metadata-only"
            if gpu_field.nbytes <= verify_max_bytes:
                cpu_bytes = os.pread(
                    fd,
                    gpu_field.nbytes,
                    file_offset + int(cpu_desc["field_rel_offset"]),
                )
                gpu_bytes = cp.asnumpy(gpu_field.array).tobytes()
                if gpu_bytes != cpu_bytes:
                    raise AssertionError(f"field bytes differ for {name!r}")
                data_result = "bytes-match"
            print(
                f"  {name}: type={gpu_field.type} shape={gpu_field.shape} "
                f"nbytes={gpu_field.nbytes} offset={gpu_field.device_offset} "
                f"{data_result}"
            )
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="one full-data .xtc2 stream")
    parser.add_argument("--detname", required=True)
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--algname", required=True)
    parser.add_argument("--field", default=None, help="verify one field (default: all)")
    parser.add_argument(
        "--nbytes",
        type=_int_arg,
        default=64 << 20,
        help="contiguous bytes to read after Configure (default: 64 MiB)",
    )
    parser.add_argument("--max-dgrams", type=int, default=4096)
    parser.add_argument(
        "--max-shapes-per-dgram",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--verify-max-bytes",
        type=_int_arg,
        default=16 << 20,
        help="maximum field size copied to CPU for byte verification",
    )
    args = parser.parse_args(argv)

    path = os.path.abspath(args.file)
    config = _read_config(path)
    config_size = int(config._size)
    schema = GpuNamesSchema.from_config(config)
    selected_schemas = schema.find_all(args.detname, args.segment, args.algname)
    selected_fields = {
        field.name
        for selected_schema in selected_schemas
        for field in selected_schema.fields
    }
    if args.field is not None and args.field not in selected_fields:
        raise KeyError(
            f"{args.detname}[{args.segment}].{args.algname} has no "
            f"field {args.field!r}"
        )

    available = max(0, os.path.getsize(path) - config_size)
    nbytes = min(int(args.nbytes), available)
    if nbytes == 0:
        raise RuntimeError("file has no bytes after Configure")
    data_gpu = _kvikio_read(path, config_size, nbytes)

    batch = GPUDgramBatch(
        data_gpu,
        schema,
        max_dgrams=args.max_dgrams,
        max_shapes_per_dgram=args.max_shapes_per_dgram,
    )
    print(
        f"file={path} configure_bytes={config_size} gpu_read_bytes={nbytes} "
        f"dgrams={len(batch)} indexed_bytes={batch.indexed_nbytes} "
        f"trailing_bytes={batch.trailing_nbytes} "
        f"index_status={STATUS_NAMES.get(batch.index_status, batch.index_status)}"
    )
    print(
        f"get={args.detname}[{args.segment}].{args.algname} "
        "names_ids="
        + ",".join(
            f"0x{selected_schema.names_id:x}"
            for selected_schema in selected_schemas
        )
    )

    found = 0
    for gpudgram in batch:
        service = SERVICE_NAMES.get(gpudgram.service, str(gpudgram.service))
        walk_status = int(batch.walk_status[gpudgram.index])
        n_shapes = int(batch.shape_counts[gpudgram.index])
        print(
            f"dgram={gpudgram.index} service={service} "
            f"timestamp={gpudgram.timestamp} shapes_data={n_shapes} "
            f"walk={STATUS_NAMES.get(walk_status, walk_status)}"
        )
        gpu_alg = gpudgram.get(args.detname, args.segment, args.algname)
        if gpu_alg is None:
            continue
        found += 1
        _verify_dgram(
            path,
            config,
            config_size,
            batch,
            gpudgram.index,
            gpu_alg,
            args.field,
            args.verify_max_bytes,
        )

    if found == 0:
        raise RuntimeError("selected detector/segment/algorithm was not found")
    print(f"PASS: verified {found} matching dgrams")


if __name__ == "__main__":
    main()
