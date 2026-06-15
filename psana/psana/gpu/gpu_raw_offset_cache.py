import os

import numpy as np


GPU_RAW_OFFSET_DTYPE = np.dtype(
    [
        ("stream_id", "<u8"),
        ("names_id_value", "<u8"),
        ("segment", "<u8"),
        ("raw_rel_offset", "<u8"),
        ("raw_nbytes", "<u8"),
        ("dim0", "<u8"),
        ("dim1", "<u8"),
        ("dim2", "<u8"),
        ("dtype_size", "<u8"),
        ("expected_bd_size", "<u8"),
    ]
)


class GpuRawOffsetCacheError(RuntimeError):
    pass


class GpuRawOffsetCache:
    """
    Cache raw payload offsets inferred from one bigdata dgram per GPU stream.

    BD can use this as a bootstrap helper: the first L1Accept seen for a GPU
    stream provides bd_offset/bd_size, and this helper reads that dgram once,
    asks psana.dgram.Dgram for targeted raw descriptors, and stores offsets
    relative to the beginning of the bigdata dgram.
    """

    def __init__(
        self,
        gpu_stream_ids,
        xtc_files=None,
        fds=None,
        configs=None,
        det_name="jungfrau",
        alg_name="raw",
        field_name="raw",
    ):
        self.gpu_stream_ids = set(parse_stream_ids(gpu_stream_ids) or ())
        self.xtc_files = list(xtc_files) if xtc_files is not None else None
        self.fds = list(fds) if fds is not None else None
        self.configs = list(configs) if configs is not None else None
        if self.configs is None:
            raise GpuRawOffsetCacheError(
                "configs are required; raw offsets are extracted through "
                "Dgram.raw_descriptors()"
            )
        self.det_name = det_name
        self.alg_name = alg_name
        self.field_name = field_name
        self._owned_fds = {}
        self._rows_by_key = {}
        self._cached_streams = set()

    def close(self):
        for fd in self._owned_fds.values():
            os.close(fd)
        self._owned_fds.clear()

    @property
    def expected_n_streams(self):
        return len(self.gpu_stream_ids)

    @property
    def n_cached_streams(self):
        return len(self._cached_streams)

    @property
    def n_rows(self):
        return len(self._rows_by_key)

    @property
    def ready(self):
        return (
            bool(self.gpu_stream_ids)
            and self._cached_streams >= self.gpu_stream_ids
        )

    def is_stream_cached(self, stream_id):
        return int(stream_id) in self._cached_streams

    def ensure_stream_cached(self, stream_id, bd_offset, bd_size):
        stream_id = int(stream_id)
        bd_offset = int(bd_offset)
        bd_size = int(bd_size)

        if stream_id not in self.gpu_stream_ids:
            return 0

        if self.is_stream_cached(stream_id):
            return len(self.rows_for_stream(stream_id))

        if bd_size <= 0:
            raise GpuRawOffsetCacheError(
                f"Cannot bootstrap stream {stream_id}: bd_size={bd_size}"
            )

        dgram_bytes = self._pread(stream_id, bd_size, bd_offset)
        rows = self._build_rows(
            dgram_bytes,
            stream_id,
            expected_bd_size=bd_size,
        )
        if rows.size == 0:
            raise GpuRawOffsetCacheError(
                f"Stream {stream_id} produced no "
                f"{self.det_name}/{self.alg_name}/{self.field_name} descriptors"
            )

        for row in rows:
            key = (int(row["stream_id"]), int(row["names_id_value"]))
            self._rows_by_key[key] = row
        self._cached_streams.add(stream_id)

        return len(rows)

    def rows_for_stream(self, stream_id):
        stream_id = int(stream_id)
        rows = [
            self._rows_by_key[key]
            for key in sorted(self._rows_by_key)
            if key[0] == stream_id
        ]
        return np.asarray(rows, dtype=GPU_RAW_OFFSET_DTYPE)

    @property
    def rows(self):
        rows = [
            self._rows_by_key[key]
            for key in sorted(self._rows_by_key)
        ]
        return np.asarray(rows, dtype=GPU_RAW_OFFSET_DTYPE)

    def _pread(self, stream_id, size, offset):
        fd = self._fd_for_stream(stream_id)
        data = os.pread(fd, size, offset)
        if len(data) != size:
            raise GpuRawOffsetCacheError(
                f"Short read for stream {stream_id}: offset={offset} "
                f"asked={size} got={len(data)}"
            )
        return data

    def _fd_for_stream(self, stream_id):
        if self.fds is not None:
            if stream_id >= len(self.fds):
                raise GpuRawOffsetCacheError(
                    f"stream_id {stream_id} outside fds length {len(self.fds)}"
                )
            return int(self.fds[stream_id])

        if self.xtc_files is None:
            raise GpuRawOffsetCacheError("xtc_files or fds are required")

        if stream_id >= len(self.xtc_files):
            raise GpuRawOffsetCacheError(
                f"stream_id {stream_id} outside xtc_files length {len(self.xtc_files)}"
            )

        fd = self._owned_fds.get(stream_id)
        if fd is None:
            fd = os.open(self.xtc_files[stream_id], os.O_RDONLY)
            self._owned_fds[stream_id] = fd
        return fd

    def _build_rows(self, dgram_bytes, stream_id, expected_bd_size):
        if stream_id >= len(self.configs):
            raise GpuRawOffsetCacheError(
                f"stream_id {stream_id} outside configs length {len(self.configs)}"
            )

        from psana import dgram

        config = self.configs[stream_id]
        pydgram = dgram.Dgram(
            view=bytearray(dgram_bytes),
            config=config,
            offset=0,
        )
        descriptors = pydgram.raw_descriptors(
            config,
            det_name=self.det_name,
            alg_name=self.alg_name,
            field_name=self.field_name,
        )
        return _build_raw_offset_rows_from_descriptors(
            descriptors,
            stream_id,
            expected_bd_size=expected_bd_size,
        )


def parse_stream_ids(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return tuple(int(stream_id) for stream_id in raw)


def _build_raw_offset_rows_from_descriptors(
    descriptors,
    stream_id,
    expected_bd_size=0,
):
    stream_id = int(stream_id)
    expected_bd_size = int(expected_bd_size)
    rows = []

    for desc in descriptors:
        names_id = int(desc["names_id_value"])
        shape = tuple(int(dim) for dim in desc["shape"])
        dim0, dim1, dim2 = shape
        raw_nbytes = int(desc["field_nbytes"])
        dtype_size = raw_nbytes // (dim0 * dim1 * dim2)
        rows.append(
            (
                stream_id,
                names_id,
                int(desc["segment"]),
                int(desc["field_rel_offset"]),
                raw_nbytes,
                dim0,
                dim1,
                dim2,
                dtype_size,
                expected_bd_size,
            )
        )

    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return np.asarray(rows, dtype=GPU_RAW_OFFSET_DTYPE)
