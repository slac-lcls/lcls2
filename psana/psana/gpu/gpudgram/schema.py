"""Compile Configure Names metadata into compact GPU parser tables."""

from dataclasses import dataclass

import numpy as np


NAMES_ID = 0
NAMES_SEGMENT = 1
NAMES_DET_KEY = 2
NAMES_ALG_KEY = 3
NAMES_FIRST_FIELD = 4
NAMES_N_FIELDS = 5
NAMES_NCOLS = 6

FIELD_KEY = 0
FIELD_TYPE = 1
FIELD_ELEMENT_SIZE = 2
FIELD_RANK = 3
FIELD_SHAPE_INDEX = 4
FIELD_NCOLS = 5

SCALAR_SHAPE_INDEX = np.iinfo(np.uint64).max


@dataclass(frozen=True)
class FieldSchema:
    name: str
    key: int
    type: int
    element_size: int
    rank: int
    shape_index: int


@dataclass(frozen=True)
class NamesSchema:
    det_name: str
    det_key: int
    det_type: str
    det_id: str
    alg_name: str
    alg_key: int
    alg_version: tuple
    segment: int
    names_id: int
    first_field: int
    fields: tuple


@dataclass(frozen=True)
class DeviceSchema:
    names: object
    fields: object


class GpuNamesSchema:
    """Run-scoped, detector-independent schema compiled from Configure."""

    def __init__(self, entries):
        entries = tuple(entries)
        det_names = sorted({str(entry["det_name"]) for entry in entries})
        alg_names = sorted({str(entry["alg_name"]) for entry in entries})
        field_names = sorted(
            {
                str(field["name"])
                for entry in entries
                for field in entry["fields"]
            }
        )
        self.det_keys = {name: key for key, name in enumerate(det_names)}
        self.alg_keys = {name: key for key, name in enumerate(alg_names)}
        self.field_keys = {name: key for key, name in enumerate(field_names)}

        # config_names() is backed by an unordered NamesLookup.  Sort here so
        # the packed schema and every downstream row are deterministic.
        entries = tuple(sorted(entries, key=lambda entry: int(entry["names_id_value"])))
        names_rows = np.empty((len(entries), NAMES_NCOLS), dtype=np.uint64)
        field_rows = []
        names = []
        by_selector = {}

        for row_index, entry in enumerate(entries):
            det_name = str(entry["det_name"])
            alg_name = str(entry["alg_name"])
            segment = int(entry["segment"])
            selector = (det_name, segment, alg_name)
            first_field = len(field_rows)
            fields = []
            for expected_index, field in enumerate(entry["fields"]):
                field_index = int(field["field_index"])
                if field_index != expected_index:
                    raise ValueError(
                        f"{selector!r} has non-consecutive field index "
                        f"{field_index}; expected {expected_index}"
                    )
                rank = int(field["rank"])
                shape_index = int(field["shape_index"])
                if (rank == 0) != (shape_index == -1):
                    raise ValueError(
                        f"{selector!r} field {field['name']!r} has "
                        f"rank={rank}, shape_index={shape_index}"
                    )
                name = str(field["name"])
                field_schema = FieldSchema(
                    name=name,
                    key=self.field_keys[name],
                    type=int(field["type"]),
                    element_size=int(field["element_size"]),
                    rank=rank,
                    shape_index=shape_index,
                )
                fields.append(field_schema)
                field_rows.append(
                    (
                        field_schema.key,
                        field_schema.type,
                        field_schema.element_size,
                        field_schema.rank,
                        SCALAR_SHAPE_INDEX if shape_index < 0 else shape_index,
                    )
                )

            names_id = int(entry["names_id_value"])
            names_rows[row_index] = (
                names_id,
                segment,
                self.det_keys[det_name],
                self.alg_keys[alg_name],
                first_field,
                len(fields),
            )
            names_schema = NamesSchema(
                det_name=det_name,
                det_key=self.det_keys[det_name],
                det_type=str(entry["det_type"]),
                det_id=str(entry["det_id"]),
                alg_name=alg_name,
                alg_key=self.alg_keys[alg_name],
                alg_version=tuple(int(value) for value in entry["alg_version"]),
                segment=segment,
                names_id=names_id,
                first_field=first_field,
                fields=tuple(fields),
            )
            names.append(names_schema)
            by_selector.setdefault(selector, []).append(names_schema)

        if field_rows:
            fields_table = np.asarray(field_rows, dtype=np.uint64).reshape(
                -1, FIELD_NCOLS
            )
        else:
            fields_table = np.empty((0, FIELD_NCOLS), dtype=np.uint64)

        self.names = tuple(names)
        self.names_table = names_rows
        self.fields_table = fields_table
        self._by_selector = {
            selector: tuple(matches) for selector, matches in by_selector.items()
        }

    @classmethod
    def from_config(cls, config):
        return cls(config.config_names())

    def find(self, det_name, segment, alg_name):
        matches = self.find_all(det_name, segment, alg_name)
        if len(matches) != 1:
            names_ids = ", ".join(f"0x{match.names_id:x}" for match in matches)
            raise ValueError(
                f"Configure selector {(str(det_name), int(segment), str(alg_name))!r} "
                f"has multiple NamesIds: {names_ids}"
            )
        return matches[0]

    def find_all(self, det_name, segment, alg_name):
        selector = (str(det_name), int(segment), str(alg_name))
        try:
            return self._by_selector[selector]
        except KeyError as exc:
            available = ", ".join(
                f"{det}[{seg}].{alg}" for det, seg, alg in sorted(self._by_selector)
            )
            raise KeyError(
                f"Configure has no {selector!r}; available: {available}"
            ) from exc

    def to_device(self, cp):
        return DeviceSchema(
            names=cp.asarray(self.names_table),
            fields=cp.asarray(self.fields_table),
        )
