import os

import numpy as np


NAME_TYPE_UINT16 = 1
JUNGFRAU_RAW_DTYPE_SIZE = 2
JUNGFRAU_RAW_RANK = 3
JUNGFRAU_RAW_SHAPE = (1, 512, 1024)
JUNGFRAU_RAW_DATA_OFFSET = 0
JUNGFRAU_RAW_FIELD_INDEX = 0
JUNGFRAU_RAW_SHAPE_INDEX = 0

GPU_JUNGFRAU_CONFIG_DTYPE = np.dtype(
    [
        ("stream_id", "<u8"),
        ("names_id_value", "<u8"),
        ("segment", "<u8"),
        ("raw_data_offset", "<u8"),
        ("dtype_size", "<u8"),
        ("dim0", "<u8"),
        ("dim1", "<u8"),
        ("dim2", "<u8"),
    ]
)


class GpuJungfrauConfigTable:
    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=GPU_JUNGFRAU_CONFIG_DTYPE)

    @property
    def n_rows(self):
        return int(self.rows.shape[0])

    def __len__(self):
        return self.n_rows

    def __bool__(self):
        return self.n_rows > 0

    def __nonzero__(self):
        return self.__bool__()


def parse_stream_ids(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return tuple(int(stream_id) for stream_id in raw)


def build_jungfrau_config_table(configs, gpu_stream_ids=None, det_name="jungfrau"):
    gpu_stream_ids = parse_stream_ids(gpu_stream_ids)
    gpu_stream_id_set = set(gpu_stream_ids) if gpu_stream_ids is not None else None

    rows = []
    for stream_id, cfg_dgram in enumerate(configs):
        if gpu_stream_id_set is not None and stream_id not in gpu_stream_id_set:
            continue

        for names in _config_names(cfg_dgram):
            if names["det_name"] != det_name or names["alg_name"] != "raw":
                continue

            if not _is_supported_raw(names["fields"]):
                continue

            rows.append(
                (
                    stream_id,
                    names["names_id_value"],
                    names["segment"],
                    JUNGFRAU_RAW_DATA_OFFSET,
                    JUNGFRAU_RAW_DTYPE_SIZE,
                    JUNGFRAU_RAW_SHAPE[0],
                    JUNGFRAU_RAW_SHAPE[1],
                    JUNGFRAU_RAW_SHAPE[2],
                )
            )

    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return GpuJungfrauConfigTable(rows)


def build_jungfrau_config_table_from_env(configs):
    return build_jungfrau_config_table(
        configs,
        gpu_stream_ids=os.environ.get("PS_TEST_GPU_STREAM_IDS"),
    )


def _config_names(cfg_dgram):
    try:
        return cfg_dgram.config_names()
    except AttributeError as exc:
        raise RuntimeError(
            "Dgram.config_names() is required to build GPU config tables"
        ) from exc


def _is_supported_raw(fields):
    for field in fields:
        if field["name"] != "raw":
            continue
        return (
            field["type"] == NAME_TYPE_UINT16
            and field["rank"] == JUNGFRAU_RAW_RANK
            and field["field_index"] == JUNGFRAU_RAW_FIELD_INDEX
            and field["shape_index"] == JUNGFRAU_RAW_SHAPE_INDEX
        )
    return False
