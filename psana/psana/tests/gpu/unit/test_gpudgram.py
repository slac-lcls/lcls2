import numpy as np
import pytest

from psana.gpu.gpudgram.config import (
    FIELD_ELEMENT_SIZE,
    FIELD_RANK,
    FIELD_SHAPE_INDEX,
    NAMES_FIRST_FIELD,
    NAMES_ID,
    NAMES_N_FIELDS,
    NAMES_STREAM_ID,
    SCALAR_SHAPE_INDEX,
    GpuFieldHandle,
    GpuStreamConfigTable,
)


class _Config:
    def __init__(self, entries):
        self._entries = entries

    def config_names(self):
        return self._entries


def _entry(det_name, segment, alg_name, names_id, fields):
    return {
        "det_name": det_name,
        "det_type": "generic",
        "det_id": f"{det_name}-id",
        "alg_name": alg_name,
        "alg_version": (1, 2, 3),
        "segment": segment,
        "names_id_value": names_id,
        "fields": fields,
    }


def _field(name, type_id, element_size, rank, field_index, shape_index):
    return {
        "name": name,
        "type": type_id,
        "element_size": element_size,
        "rank": rank,
        "field_index": field_index,
        "shape_index": shape_index,
    }


def test_config_table_is_stream_indexed_and_deterministic():
    configs = GpuStreamConfigTable.from_configs(
        [
            _Config(
                [
                    _entry(
                        "det_b",
                        3,
                        "raw",
                        20,
                        [_field("pixels", 1, 2, 2, 0, 0)],
                    ),
                    _entry(
                        "det_a",
                        0,
                        "fex",
                        10,
                        [
                            _field("energy", 8, 4, 0, 0, -1),
                            _field("waveform", 5, 2, 1, 1, 0),
                        ],
                    ),
                ]
            ),
            _Config([_entry("det_c", 1, "raw", 10, [])]),
        ]
    )

    assert configs.stream_names_index.tolist() == [0, 2, 3]
    assert configs.names_table[:, NAMES_STREAM_ID].tolist() == [0, 0, 1]
    assert configs.names_table[:, NAMES_ID].tolist() == [10, 20, 10]
    assert configs.names_table[:, NAMES_FIRST_FIELD].tolist() == [0, 2, 3]
    assert configs.names_table[:, NAMES_N_FIELDS].tolist() == [2, 1, 0]
    assert configs.fields_table[:, FIELD_ELEMENT_SIZE].tolist() == [4, 2, 2]
    assert configs.fields_table[:, FIELD_RANK].tolist() == [0, 1, 2]
    assert configs.fields_table[0, FIELD_SHAPE_INDEX] == SCALAR_SHAPE_INDEX
    assert configs.fields_table[1:, FIELD_SHAPE_INDEX].tolist() == [0, 0]
    assert configs.det_keys == {"det_a": 0, "det_b": 1, "det_c": 2}
    assert configs.alg_keys == {"fex": 0, "raw": 1}
    assert configs.names_for_id(1, 10).det_name == "det_c"


def test_field_selector_resolves_to_numeric_device_handle():
    configs = GpuStreamConfigTable(
        {
            0: [_entry("other", 0, "raw", 10, [])],
            1: [
                _entry(
                    "det",
                    2,
                    "raw",
                    0x10C,
                    [
                        _field("counter", 3, 8, 0, 0, -1),
                        _field("array", 1, 2, 2, 1, 0),
                    ],
                )
            ],
        }
    )

    handle = configs.resolve("det", 2, "raw", "array")
    assert isinstance(handle, GpuFieldHandle)
    assert handle == GpuFieldHandle(
        stream_id=1,
        names_id=0x10C,
        config_names_index=1,
        config_field_index=1,
        field_index=1,
        type=1,
        element_size=2,
        rank=2,
        shape_index=0,
    )
    assert configs[1].resolve("det", 2, "raw", "array") == handle


def test_resolve_all_preserves_detector_ownership_across_streams():
    entry = _entry(
        "det", 0, "raw", 10, [_field("array", 1, 2, 1, 0, 0)]
    )
    configs = GpuStreamConfigTable({0: [entry], 1: [entry]})

    handles = configs.resolve_all("det", 0, "raw", "array")
    assert [handle.stream_id for handle in handles] == [0, 1]
    with pytest.raises(ValueError, match="multiple matches"):
        configs.resolve("det", 0, "raw", "array")


@pytest.mark.parametrize(
    "fields, match",
    [
        ([_field("x", 2, 4, 0, 1, -1)], "field index"),
        ([_field("x", 2, 4, 0, 0, 0)], "shape_index"),
        ([_field("x", 2, 4, 1, 0, -1)], "shape_index"),
    ],
)
def test_config_table_rejects_inconsistent_field_metadata(fields, match):
    with pytest.raises(ValueError, match=match):
        GpuStreamConfigTable({0: [_entry("det", 0, "raw", 10, fields)]})


def test_config_table_rejects_duplicate_names_id_within_stream():
    with pytest.raises(ValueError, match="duplicate Configure NamesId"):
        GpuStreamConfigTable(
            {
                0: [
                    _entry("det", 0, "raw", 10, []),
                    _entry("other", 0, "raw", 10, []),
                ]
            }
        )


def test_to_device_records_host_counts_without_copying_strings():
    configs = GpuStreamConfigTable(
        {
            0: [
                _entry(
                    "det",
                    0,
                    "raw",
                    10,
                    [_field("array", 1, 2, 1, 0, 0)],
                )
            ],
            1: [],
        }
    )
    device = configs.to_device(np)

    assert device.n_streams == 2
    assert device.n_names == 1
    assert device.n_fields == 1
    assert device.stream_names_index.dtype == np.uint64
    assert device.names.shape == (1, 7)
    assert device.fields.shape == (1, 5)
