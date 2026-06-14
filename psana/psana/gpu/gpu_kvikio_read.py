from dataclasses import dataclass

import numpy as np

from psana.gpu.gpu_compare import digest_bytes


# Bigdata read descriptor table columns.  This table is intentionally plain
# uint64 so it can be joined with CPU-built GPU work descriptor tables.
DESC_EVENT_INDEX = 0
DESC_STREAM_ID = 1
DESC_TIMESTAMP = 2
DESC_FILE_OFFSET = 3
DESC_READ_SIZE = 4
DESC_DEVICE_OFFSET = 5
DESC_NCOLS = 6


@dataclass
class KvikioBatchRead:
    by_timestamp: dict
    read_descs: tuple
    desc_table: np.ndarray
    data_gpu: object = None


class KvikioGpuReader:
    def __init__(self, task_size=None, keep_device_buffers=False):
        import cupy as cp
        import kvikio

        self.cp = cp
        self.kvikio = kvikio
        self.task_size = task_size
        self.keep_device_buffers = keep_device_buffers
        self._files = {}

    def close(self):
        for fh in self._files.values():
            fh.close()
        self._files.clear()

    def read_batch(self, gpu_view, bd_dm):
        read_descs = tuple(gpu_view.iter_read_descs(bd_dm))
        desc_table = self._build_desc_table(read_descs)

        if not read_descs:
            return KvikioBatchRead({}, read_descs, desc_table)

        total_nbytes = int(
            desc_table[-1, DESC_DEVICE_OFFSET] + desc_table[-1, DESC_READ_SIZE]
        )
        data_gpu = self.cp.empty(total_nbytes, dtype=self.cp.uint8)

        futures = []
        by_timestamp = {}
        for desc, row in zip(read_descs, desc_table):
            read_size = int(row[DESC_READ_SIZE])
            device_offset = int(row[DESC_DEVICE_OFFSET])
            per_stream = by_timestamp.setdefault(desc.timestamp, {})

            if read_size == 0:
                per_stream[desc.stream_id] = (0, digest_bytes(b""))
                continue

            cu_file = self._file_for_stream(bd_dm, desc.stream_id)
            dst = data_gpu[device_offset:device_offset + read_size]
            future = cu_file.pread(
                dst,
                size=read_size,
                file_offset=int(row[DESC_FILE_OFFSET]),
                task_size=self.task_size,
            )
            futures.append((desc, device_offset, read_size, future))

        for desc, device_offset, read_size, future in futures:
            nread = int(future.get())
            if nread != read_size:
                raise RuntimeError(
                    f"KvikIO GPU read failed: event={desc.batch_event_index} "
                    f"stream={desc.stream_id} offset={desc.offset} "
                    f"asked={read_size} got={nread}"
                )

            # D2H validation path.  Later kernels can consume data_gpu directly.
            host = data_gpu[device_offset:device_offset + read_size].get()
            per_stream = by_timestamp.setdefault(desc.timestamp, {})
            per_stream[desc.stream_id] = (read_size, digest_bytes(host))

        if self.keep_device_buffers:
            return KvikioBatchRead(
                by_timestamp,
                read_descs,
                desc_table,
                data_gpu=data_gpu,
            )

        return KvikioBatchRead(by_timestamp, read_descs, desc_table)

    def _file_for_stream(self, bd_dm, stream_id):
        cu_file = self._files.get(stream_id)
        if cu_file is None:
            cu_file = self.kvikio.CuFile(str(bd_dm.xtc_files[stream_id]), "r")
            self._files[stream_id] = cu_file
        return cu_file

    @staticmethod
    def _build_desc_table(read_descs):
        desc_table = np.empty((len(read_descs), DESC_NCOLS), dtype=np.uint64)

        device_offset = 0
        for row, desc in zip(desc_table, read_descs):
            row[DESC_EVENT_INDEX] = desc.batch_event_index
            row[DESC_STREAM_ID] = desc.stream_id
            row[DESC_TIMESTAMP] = desc.timestamp
            row[DESC_FILE_OFFSET] = desc.offset
            row[DESC_READ_SIZE] = desc.size
            row[DESC_DEVICE_OFFSET] = device_offset
            device_offset += desc.size

        return desc_table
