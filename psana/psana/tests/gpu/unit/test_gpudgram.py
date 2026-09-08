import numpy as np
import pytest

from psana.gpu.gpudgram.schema import (
    FIELD_ELEMENT_SIZE,
    FIELD_RANK,
    FIELD_SHAPE_INDEX,
    NAMES_FIRST_FIELD,
    NAMES_ID,
    NAMES_N_FIELDS,
    SCALAR_SHAPE_INDEX,
    GpuNamesSchema,
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


def test_schema_is_detector_independent_and_deterministic():
    entries = [
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
    schema = GpuNamesSchema.from_config(_Config(entries))

    # Input order follows an unordered Configure lookup; packed order does not.
    assert schema.names_table[:, NAMES_ID].tolist() == [10, 20]
    fex = schema.find("det_a", 0, "fex")
    raw = schema.find("det_b", 3, "raw")
    assert fex.names_id == 10
    assert raw.names_id == 20
    assert schema.names_table[0, NAMES_FIRST_FIELD] == 0
    assert schema.names_table[0, NAMES_N_FIELDS] == 2
    assert schema.names_table[1, NAMES_FIRST_FIELD] == 2
    assert schema.names_table[1, NAMES_N_FIELDS] == 1

    fields = schema.fields_table
    assert fields[:, FIELD_ELEMENT_SIZE].tolist() == [4, 2, 2]
    assert fields[:, FIELD_RANK].tolist() == [0, 1, 2]
    assert fields[0, FIELD_SHAPE_INDEX] == SCALAR_SHAPE_INDEX
    assert fields[1:, FIELD_SHAPE_INDEX].tolist() == [0, 0]
    assert schema.det_keys == {"det_a": 0, "det_b": 1}
    assert schema.alg_keys == {"fex": 0, "raw": 1}


def test_schema_preserves_multiple_names_ids_for_one_selector():
    entries = [
        _entry("det", 0, "raw", 10, []),
        _entry("det", 0, "raw", 11, []),
    ]
    schema = GpuNamesSchema(entries)
    assert [entry.names_id for entry in schema.find_all("det", 0, "raw")] == [10, 11]
    with pytest.raises(ValueError, match="multiple NamesIds"):
        schema.find("det", 0, "raw")


@pytest.mark.parametrize(
    "fields, match",
    [
        ([_field("x", 2, 4, 0, 1, -1)], "field index"),
        ([_field("x", 2, 4, 0, 0, 0)], "shape_index"),
        ([_field("x", 2, 4, 1, 0, -1)], "shape_index"),
    ],
)
def test_schema_rejects_inconsistent_field_metadata(fields, match):
    with pytest.raises(ValueError, match=match):
        GpuNamesSchema([_entry("det", 0, "raw", 10, fields)])


def test_find_reports_available_selectors():
    schema = GpuNamesSchema(
        [_entry("det", 2, "raw", 10, [_field("x", 2, 4, 0, 0, -1)])]
    )
    with pytest.raises(KeyError, match=r"det\[2\]\.raw"):
        schema.find("missing", 0, "raw")
