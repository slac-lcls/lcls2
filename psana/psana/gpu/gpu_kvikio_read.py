from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from psana.gpu.gpu_compare import digest_bytes


# Device-side descriptor table columns.  This table is intentionally plain uint64
# so it can be copied to the GPU as-is and later consumed by kernels.
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
    desc_table_gpu: object = None
    data_gpu: object = None


@dataclass
class PendingBatch:
    """Holds in-flight KvikIO pread futures and the GPU-side buffers they write into.

    Returned by ``KvikioGpuReader.issue_batch()``.  Pass to ``wait_batch()``
    once the caller has done other work (e.g. CPU EventManager path) to collect
    the completed reads.
    """
    gpu_view: object            # GpuBatchView — needed by wait_batch callers
    read_descs: tuple
    desc_table: np.ndarray
    desc_table_gpu: object      # cp.ndarray uint64
    data_gpu: object            # cp.ndarray uint8  (reads landing here)
    futures: List[Tuple]        # [(desc, device_offset, read_size, kvikio_future)]


class KvikioGpuReader:
    def __init__(self, task_size=None):
        import cupy as cp
        import kvikio

        self.cp = cp
        self.kvikio = kvikio
        self.task_size = task_size
        self._files = {}

    def close(self):
        for fh in self._files.values():
            fh.close()
        self._files.clear()

    def issue_batch(self, gpu_view, bd_dm) -> "PendingBatch":
        """Issue GDS reads for a GPU batch non-blocking.

        All KvikIO pread() calls are issued immediately and return futures.
        The caller can do other work (e.g. CPU EventManager path) before
        calling wait_batch() to collect the completed reads.

        Parameters
        ----------
        gpu_view : GpuBatchView describing the batch
        bd_dm    : DgramManager holding bigdata file descriptors

        Returns
        -------
        PendingBatch with in-flight futures.  Pass to wait_batch().
        """
        read_descs = tuple(gpu_view.iter_read_descs(bd_dm))
        desc_table = self._build_desc_table(read_descs)

        if not read_descs:
            return PendingBatch(
                gpu_view=gpu_view,
                read_descs=read_descs,
                desc_table=desc_table,
                desc_table_gpu=self.cp.empty((0, DESC_NCOLS), dtype=self.cp.uint64),
                data_gpu=self.cp.empty(0, dtype=self.cp.uint8),
                futures=[],
            )

        desc_table_gpu = self.cp.asarray(desc_table)
        total_nbytes = int(
            desc_table[-1, DESC_DEVICE_OFFSET] + desc_table[-1, DESC_READ_SIZE]
        )
        data_gpu = self.cp.empty(total_nbytes, dtype=self.cp.uint8)

        futures = []
        for desc, row in zip(read_descs, desc_table):
            read_size = int(row[DESC_READ_SIZE])
            if read_size == 0:
                continue
            device_offset = int(row[DESC_DEVICE_OFFSET])
            cu_file = self._file_for_stream(bd_dm, desc.stream_id)
            dst = data_gpu[device_offset:device_offset + read_size]
            future = cu_file.pread(
                dst,
                size=read_size,
                file_offset=int(row[DESC_FILE_OFFSET]),
                task_size=self.task_size,
            )
            futures.append((desc, device_offset, read_size, future))

        return PendingBatch(
            gpu_view=gpu_view,
            read_descs=read_descs,
            desc_table=desc_table,
            desc_table_gpu=desc_table_gpu,
            data_gpu=data_gpu,
            futures=futures,
        )

    def wait_batch(self, pending: "PendingBatch",
                   compute_digest: bool = False) -> KvikioBatchRead:
        """Wait for in-flight reads from issue_batch() and return a KvikioBatchRead.

        Parameters
        ----------
        pending        : PendingBatch returned by issue_batch()
        compute_digest : bool, default False
            When True perform a D2H copy per descriptor and record digests in
            by_timestamp.  Only needed by --compare-nosplit validation mode.

        Returns
        -------
        KvikioBatchRead with data_gpu and desc_table_gpu populated.
        """
        if not pending.futures:
            return KvikioBatchRead(
                {}, pending.read_descs, pending.desc_table,
                desc_table_gpu=pending.desc_table_gpu,
                data_gpu=pending.data_gpu,
            )

        by_timestamp = {}
        for desc, device_offset, read_size, future in pending.futures:
            nread = int(future.get())
            if nread != read_size:
                raise RuntimeError(
                    f"KvikIO GPU read failed: event={desc.batch_event_index} "
                    f"stream={desc.stream_id} offset={desc.offset} "
                    f"asked={read_size} got={nread}"
                )
            if compute_digest:
                host = pending.data_gpu[device_offset:device_offset + read_size].get()
                by_timestamp.setdefault(desc.timestamp, {})[desc.stream_id] = (
                    read_size, digest_bytes(host)
                )

        # Fill zero-size entries for compute_digest mode.
        if compute_digest:
            for desc, row in zip(pending.read_descs, pending.desc_table):
                if int(row[DESC_READ_SIZE]) == 0:
                    by_timestamp.setdefault(desc.timestamp, {})[desc.stream_id] = (
                        0, digest_bytes(b"")
                    )

        return KvikioBatchRead(
            by_timestamp,
            pending.read_descs,
            pending.desc_table,
            desc_table_gpu=pending.desc_table_gpu,
            data_gpu=pending.data_gpu,
        )

    def read_batch(self, gpu_view, bd_dm,
                   compute_digest: bool = False) -> KvikioBatchRead:
        """Issue and immediately wait for all GDS reads (sequential convenience).

        Equivalent to ``wait_batch(issue_batch(gpu_view, bd_dm), compute_digest)``.
        Use issue_batch() + wait_batch() directly when you want to overlap reads
        with other work (CPU EventManager path, prior-batch computation, etc.).
        """
        return self.wait_batch(self.issue_batch(gpu_view, bd_dm), compute_digest)

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
