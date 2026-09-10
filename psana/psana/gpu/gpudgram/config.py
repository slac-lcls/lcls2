"""Compile stream-indexed Configure metadata for GPU XTC parsing.

The tables in this module are derived from ``Run.configs``.  They contain the
Configure Names and field definitions needed to interpret event ShapesData;
they do not retain the raw Configure dgram bytes.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


NAMES_STREAM_ID = 0
NAMES_ID = 1
NAMES_SEGMENT = 2
NAMES_DET_KEY = 3
NAMES_ALG_KEY = 4
NAMES_FIRST_FIELD = 5
NAMES_N_FIELDS = 6
NAMES_NCOLS = 7

FIELD_KEY = 0
FIELD_TYPE = 1
FIELD_ELEMENT_SIZE = 2
FIELD_RANK = 3
FIELD_SHAPE_INDEX = 4
FIELD_NCOLS = 5

SCALAR_SHAPE_INDEX = np.iinfo(np.uint64).max


@dataclass(frozen=True)
class ConfigField:
    """One field definition from a Configure Names record."""

    name: str
    key: int
    type: int
    element_size: int
    rank: int
    shape_index: int
    config_field_index: int


@dataclass(frozen=True)
class ConfigNames:
    """One Configure Names record, including its owning stream."""

    stream_id: int
    det_name: str
    det_key: int
    det_type: str
    det_id: str
    alg_name: str
    alg_key: int
    alg_version: tuple
    segment: int
    names_id: int
    config_names_index: int
    first_field: int
    fields: tuple


@dataclass(frozen=True)
class GpuFieldHandle:
    """Numeric run-scoped selector consumed by GPU field-locator kernels."""

    stream_id: int
    names_id: int
    config_names_index: int
    config_field_index: int
    field_index: int
    type: int
    element_size: int
    rank: int
    shape_index: int


@dataclass(frozen=True)
class DeviceConfigTables:
    """Numeric Configure tables uploaded once for GPU event decoding."""

    n_streams: int
    n_names: int
    n_fields: int
    stream_names_index: object
    names: object
    fields: object


class GpuStreamConfigView:
    """View of one stream in a :class:`GpuStreamConfigTable`."""

    def __init__(self, configs, stream_id):
        self._configs = configs
        self.stream_id = int(stream_id)

    @property
    def names(self):
        begin = int(self._configs.stream_names_index[self.stream_id])
        end = int(self._configs.stream_names_index[self.stream_id + 1])
        return self._configs.names[begin:end]

    def find(self, det_name, segment, alg_name):
        return self._configs.find(
            det_name, segment, alg_name, stream_id=self.stream_id
        )

    def find_all(self, det_name, segment, alg_name):
        return self._configs.find_all(
            det_name, segment, alg_name, stream_id=self.stream_id
        )

    def find_field(self, det_name, segment, alg_name, field_name):
        return self._configs.find_field(
            det_name,
            segment,
            alg_name,
            field_name,
            stream_id=self.stream_id,
        )

    def find_fields(self, det_name, segment, alg_name, field_name):
        return self._configs.find_fields(
            det_name,
            segment,
            alg_name,
            field_name,
            stream_id=self.stream_id,
        )

    def resolve(self, det_name, segment, alg_name, field_name):
        return self._configs.resolve(
            det_name,
            segment,
            alg_name,
            field_name,
            stream_id=self.stream_id,
        )

    def resolve_all(self, det_name, segment, alg_name, field_name):
        return self._configs.resolve_all(
            det_name,
            segment,
            alg_name,
            field_name,
            stream_id=self.stream_id,
        )


class GpuStreamConfigTable:
    """Stream-indexed GPU representation compiled from ``Run.configs``.

    Detector, algorithm, and field keys are allocated globally across every
    stream so numeric selectors have one meaning for the entire run.  A Names
    record is identified by ``(stream_id, names_id)``; NamesId alone is not
    assumed to be unique across independent stream Configures.

    Parameters
    ----------
    entries_by_stream : mapping[int, iterable[dict]]
        Output from ``config.config_names()`` grouped by stream id.
    n_streams : int, optional
        Total stream count.  Required only when trailing streams have no Names
        records and therefore cannot be inferred from the mapping keys.
    """

    def __init__(self, entries_by_stream, *, n_streams=None):
        if not isinstance(entries_by_stream, Mapping):
            raise TypeError("entries_by_stream must map stream id to Configure entries")

        normalized = {}
        for stream_id, entries in entries_by_stream.items():
            stream_id = int(stream_id)
            if stream_id < 0:
                raise ValueError("stream ids must be non-negative")
            normalized[stream_id] = tuple(entries)

        inferred_n_streams = max(normalized, default=-1) + 1
        if n_streams is None:
            n_streams = inferred_n_streams
        self.n_streams = int(n_streams)
        if self.n_streams < inferred_n_streams or self.n_streams < 0:
            raise ValueError(
                f"n_streams={self.n_streams} does not cover stream ids "
                f"{sorted(normalized)}"
            )
        self.stream_ids = tuple(range(self.n_streams))

        all_entries = [
            entry
            for stream_id in self.stream_ids
            for entry in normalized.get(stream_id, ())
        ]
        det_names = sorted({str(entry["det_name"]) for entry in all_entries})
        alg_names = sorted({str(entry["alg_name"]) for entry in all_entries})
        field_names = sorted(
            {
                str(field["name"])
                for entry in all_entries
                for field in entry["fields"]
            }
        )
        self.det_keys = {name: key for key, name in enumerate(det_names)}
        self.alg_keys = {name: key for key, name in enumerate(alg_names)}
        self.field_keys = {name: key for key, name in enumerate(field_names)}

        names_rows = np.empty((len(all_entries), NAMES_NCOLS), dtype=np.uint64)
        stream_names_index = np.zeros(self.n_streams + 1, dtype=np.uint64)
        field_rows = []
        names = []
        by_selector = {}
        by_stream_selector = {}
        by_stream_names_id = {}

        row_index = 0
        for stream_id in self.stream_ids:
            stream_names_index[stream_id] = row_index
            entries = sorted(
                normalized.get(stream_id, ()),
                key=lambda entry: int(entry["names_id_value"]),
            )
            for entry in entries:
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
                            f"stream {stream_id} {selector!r} has non-consecutive "
                            f"field index {field_index}; expected {expected_index}"
                        )
                    rank = int(field["rank"])
                    shape_index = int(field["shape_index"])
                    if (rank == 0) != (shape_index == -1):
                        raise ValueError(
                            f"stream {stream_id} {selector!r} field "
                            f"{field['name']!r} has rank={rank}, "
                            f"shape_index={shape_index}"
                        )
                    name = str(field["name"])
                    field_config = ConfigField(
                        name=name,
                        key=self.field_keys[name],
                        type=int(field["type"]),
                        element_size=int(field["element_size"]),
                        rank=rank,
                        shape_index=shape_index,
                        config_field_index=len(field_rows),
                    )
                    fields.append(field_config)
                    field_rows.append(
                        (
                            field_config.key,
                            field_config.type,
                            field_config.element_size,
                            field_config.rank,
                            SCALAR_SHAPE_INDEX
                            if shape_index < 0
                            else shape_index,
                        )
                    )

                names_id = int(entry["names_id_value"])
                names_key = (stream_id, names_id)
                if names_key in by_stream_names_id:
                    raise ValueError(
                        f"duplicate Configure NamesId 0x{names_id:x} in "
                        f"stream {stream_id}"
                    )
                names_rows[row_index] = (
                    stream_id,
                    names_id,
                    segment,
                    self.det_keys[det_name],
                    self.alg_keys[alg_name],
                    first_field,
                    len(fields),
                )
                names_config = ConfigNames(
                    stream_id=stream_id,
                    det_name=det_name,
                    det_key=self.det_keys[det_name],
                    det_type=str(entry["det_type"]),
                    det_id=str(entry["det_id"]),
                    alg_name=alg_name,
                    alg_key=self.alg_keys[alg_name],
                    alg_version=tuple(int(value) for value in entry["alg_version"]),
                    segment=segment,
                    names_id=names_id,
                    config_names_index=row_index,
                    first_field=first_field,
                    fields=tuple(fields),
                )
                names.append(names_config)
                by_selector.setdefault(selector, []).append(names_config)
                by_stream_selector.setdefault(
                    (stream_id,) + selector, []
                ).append(names_config)
                by_stream_names_id[names_key] = names_config
                row_index += 1

            stream_names_index[stream_id + 1] = row_index

        if field_rows:
            fields_table = np.asarray(field_rows, dtype=np.uint64).reshape(
                -1, FIELD_NCOLS
            )
        else:
            fields_table = np.empty((0, FIELD_NCOLS), dtype=np.uint64)

        self.names = tuple(names)
        self.names_table = names_rows
        self.fields_table = fields_table
        self.stream_names_index = stream_names_index
        self._by_selector = {
            selector: tuple(matches) for selector, matches in by_selector.items()
        }
        self._by_stream_selector = {
            selector: tuple(matches)
            for selector, matches in by_stream_selector.items()
        }
        self._by_stream_names_id = by_stream_names_id

    @classmethod
    def from_configs(cls, configs):
        configs = tuple(configs)
        return cls(
            {
                stream_id: config.config_names()
                for stream_id, config in enumerate(configs)
            },
            n_streams=len(configs),
        )

    @classmethod
    def from_config(cls, config, *, stream_id=0):
        stream_id = int(stream_id)
        return cls(
            {stream_id: config.config_names()},
            n_streams=stream_id + 1,
        )

    def __len__(self):
        return self.n_streams

    def __getitem__(self, stream_id):
        stream_id = int(stream_id)
        if stream_id < 0:
            stream_id += self.n_streams
        if stream_id < 0 or stream_id >= self.n_streams:
            raise IndexError(stream_id)
        return GpuStreamConfigView(self, stream_id)

    def find(self, det_name, segment, alg_name, *, stream_id=None):
        matches = self.find_all(
            det_name, segment, alg_name, stream_id=stream_id
        )
        if len(matches) != 1:
            identities = ", ".join(
                f"stream {match.stream_id}:0x{match.names_id:x}"
                for match in matches
            )
            raise ValueError(
                f"Configure selector "
                f"{(str(det_name), int(segment), str(alg_name))!r} has "
                f"multiple matches: {identities}"
            )
        return matches[0]

    def find_all(self, det_name, segment, alg_name, *, stream_id=None):
        matches = self.matches(
            det_name, segment, alg_name, stream_id=stream_id
        )
        if matches:
            return matches
        selector = (str(det_name), int(segment), str(alg_name))
        available = ", ".join(
            f"stream {entry.stream_id}:"
            f"{entry.det_name}[{entry.segment}].{entry.alg_name}"
            for entry in self.names
            if stream_id is None or entry.stream_id == int(stream_id)
        )
        scope = "Configure streams" if stream_id is None else f"stream {stream_id}"
        raise KeyError(f"{scope} have no {selector!r}; available: {available}")

    def matches(self, det_name, segment, alg_name, *, stream_id=None):
        """Return zero or more matching ConfigNames without raising."""
        selector = (str(det_name), int(segment), str(alg_name))
        if stream_id is None:
            lookup = self._by_selector
            key = selector
        else:
            stream_id = int(stream_id)
            if stream_id < 0 or stream_id >= self.n_streams:
                raise IndexError(stream_id)
            lookup = self._by_stream_selector
            key = (stream_id,) + selector
        return lookup.get(key, ())

    def find_fields(
        self,
        det_name,
        segment,
        alg_name,
        field_name,
        *,
        stream_id=None,
    ):
        field_name = str(field_name)
        matches = []
        for names in self.find_all(
            det_name, segment, alg_name, stream_id=stream_id
        ):
            for field in names.fields:
                if field.name == field_name:
                    matches.append((names, field))
                    break
        if not matches:
            raise KeyError(
                f"{det_name}[{int(segment)}].{alg_name} has no field "
                f"{field_name!r}"
            )
        return tuple(matches)

    def find_field(
        self,
        det_name,
        segment,
        alg_name,
        field_name,
        *,
        stream_id=None,
    ):
        matches = self.find_fields(
            det_name,
            segment,
            alg_name,
            field_name,
            stream_id=stream_id,
        )
        if len(matches) != 1:
            identities = ", ".join(
                f"stream {names.stream_id}:0x{names.names_id:x}"
                for names, _ in matches
            )
            raise ValueError(
                f"field selector "
                f"{(str(det_name), int(segment), str(alg_name), str(field_name))!r} "
                f"has multiple matches: {identities}"
            )
        return matches[0]

    def resolve_all(
        self,
        det_name,
        segment,
        alg_name,
        field_name,
        *,
        stream_id=None,
    ):
        """Resolve a string selector to one numeric handle per owning stream."""
        handles = []
        for names, field in self.find_fields(
            det_name,
            segment,
            alg_name,
            field_name,
            stream_id=stream_id,
        ):
            handles.append(self._field_handle(names, field))
        return tuple(handles)

    def field_handles(
        self,
        *,
        det_names=None,
        stream_ids=None,
        arrays_only=False,
    ):
        """Return deterministic handles matching optional run-level filters."""
        det_names = (
            None if det_names is None else {str(name) for name in det_names}
        )
        stream_ids = (
            None
            if stream_ids is None
            else {int(stream_id) for stream_id in stream_ids}
        )
        handles = []
        for names in self.names:
            if det_names is not None and names.det_name not in det_names:
                continue
            if stream_ids is not None and names.stream_id not in stream_ids:
                continue
            for field in names.fields:
                if arrays_only and field.rank == 0:
                    continue
                handles.append(self._field_handle(names, field))
        return tuple(handles)

    def detector_array_handles(
        self,
        det_name,
        *,
        stream_segments,
        alg_names,
        element_size=None,
    ):
        """Resolve the primary event array for each routed detector segment.

        This is the run-setup adapter used by the integrated GPU detector
        path.  The parser itself remains fully general and can resolve any
        named field.  A detector processor, however, needs one unambiguous
        array payload per canonical segment.  Configure supplies the stream,
        segment, algorithm, field type, and rank needed to establish that
        mapping without inspecting an L1Accept dgram on the CPU.

        ``stream_segments`` maps stream ids to the physical segment ids owned
        by that stream.  ``alg_names`` limits candidates to the detector
        algorithms exposed by the selected detector interface.  Ambiguous
        layouts must be selected explicitly by a future detector adapter;
        guessing a field here would recreate detector-specific raw addressing.
        """
        det_name = str(det_name)
        alg_names = {str(name) for name in alg_names}
        if not alg_names:
            raise ValueError("alg_names must contain at least one algorithm")
        if element_size is not None:
            element_size = int(element_size)

        handles = {}
        for stream_id in sorted(stream_segments):
            stream_id = int(stream_id)
            for segment in stream_segments[stream_id]:
                segment = int(segment)
                candidates = []
                for names in self.names:
                    if (
                        names.stream_id != stream_id
                        or names.det_name != det_name
                        or names.segment != segment
                        or names.alg_name not in alg_names
                    ):
                        continue
                    for field in names.fields:
                        if field.rank == 0:
                            continue
                        if (
                            element_size is not None
                            and field.element_size != element_size
                        ):
                            continue
                        candidates.append(
                            (names, field, self._field_handle(names, field))
                        )

                if len(candidates) != 1:
                    descriptions = ", ".join(
                        f"{names.alg_name}.{field.name}"
                        f"(type={field.type},rank={field.rank},"
                        f"element_size={field.element_size})"
                        for names, field, _ in candidates
                    ) or "none"
                    raise ValueError(
                        f"Configure must identify exactly one event array for "
                        f"{det_name}[{segment}] in stream {stream_id}; "
                        f"candidates: {descriptions}"
                    )
                if segment in handles:
                    raise ValueError(
                        f"segment {det_name}[{segment}] is owned by more than "
                        "one routed stream"
                    )
                handles[segment] = candidates[0][2]
        return handles

    @staticmethod
    def _field_handle(names, field):
        return GpuFieldHandle(
            stream_id=names.stream_id,
            names_id=names.names_id,
            config_names_index=names.config_names_index,
            config_field_index=field.config_field_index,
            field_index=field.config_field_index - names.first_field,
            type=field.type,
            element_size=field.element_size,
            rank=field.rank,
            shape_index=field.shape_index,
        )

    def resolve(
        self,
        det_name,
        segment,
        alg_name,
        field_name,
        *,
        stream_id=None,
    ):
        """Resolve a string selector that must identify exactly one stream."""
        handles = self.resolve_all(
            det_name,
            segment,
            alg_name,
            field_name,
            stream_id=stream_id,
        )
        if len(handles) != 1:
            identities = ", ".join(
                f"stream {handle.stream_id}:0x{handle.names_id:x}"
                for handle in handles
            )
            raise ValueError(
                f"field selector "
                f"{(str(det_name), int(segment), str(alg_name), str(field_name))!r} "
                f"has multiple matches: {identities}"
            )
        return handles[0]

    def names_for_id(self, stream_id, names_id):
        key = (int(stream_id), int(names_id))
        try:
            return self._by_stream_names_id[key]
        except KeyError as exc:
            raise KeyError(
                f"Configure has no NamesId 0x{key[1]:x} in stream {key[0]}"
            ) from exc

    def to_device(self, cp):
        return DeviceConfigTables(
            n_streams=self.n_streams,
            n_names=len(self.names),
            n_fields=len(self.fields_table),
            stream_names_index=cp.asarray(self.stream_names_index),
            names=cp.asarray(self.names_table),
            fields=cp.asarray(self.fields_table),
        )
