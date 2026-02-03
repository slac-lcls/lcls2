import json
import logging
import numbers
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from psana import dgram
from psana import utils
from psana.dgramedit import AlgDef, DetectorDef, DgramEdit
from psana.psexp import TransitionId



def load_calib_pickle(path: str = "/dev/shm/calibconst.pkl") -> Optional[dict]:
    """
    Load a calibration pickle file if it exists.

    Args:
        path (str): Path to calibconst.pkl

    Returns:
        dict | None: Parsed pickle contents or None if missing.
    """
    try:
        with open(path, "rb") as f:
            import pickle

            return pickle.load(f)
    except FileNotFoundError:
        return None


def write_xtc_file(path: str, config_bytes: bytes, data_bytes: bytes) -> None:
    """
    Write configure+dgram bytes to disk sequentially.
    """
    with open(path, "wb") as fd:
        fd.write(config_bytes)
        fd.write(data_bytes)


class CalibXtcConverter:
    """
    Converts calibration dictionaries (from calibconst.pkl) into xtc2 dgrams.
    """

    def __init__(self, det_info: Optional[Dict[str, str]] = None):
        self.det_info = det_info or {}
        self.alg = AlgDef("calib", 0, 0, 0)
        self.logger = utils.get_logger(name="calib_xtc")


    def _extract_value(self, raw_entry: Any) -> Tuple[Any, Optional[dict]]:
        """
        Split entry into (value, metadata). Entries may be tuples (value, meta).
        """
        meta = None
        value = raw_entry
        if isinstance(raw_entry, (tuple, list)) and raw_entry:
            value = raw_entry[0]
            if len(raw_entry) > 1 and isinstance(raw_entry[1], dict):
                meta = raw_entry[1]
        return value, meta

    @staticmethod
    def _is_valid_array(value: Any) -> bool:
        return isinstance(value, np.ndarray)

    def _coerce_scalar(self, value: Any) -> Tuple[Any, type]:
        if isinstance(value, np.generic):
            scalar = value.item()
            dtype_type = value.dtype.type
        elif isinstance(value, numbers.Number):
            scalar = value
            dtype_type = np.array(value).dtype.type
        else:
            raise TypeError("Unsupported scalar type")
        return scalar, dtype_type

    def _build_datadef(
        self, det_name: str, calib_entries: Dict[str, Any]
    ) -> Tuple[Dict[str, Tuple[Any, int]], Dict[str, Any]]:
        """
        Build datadef dictionary and map of field->value to populate later.
        """
        datadef: Dict[str, Tuple[Any, int]] = {}
        field_values: Dict[str, Any] = {}

        for field, raw_entry in calib_entries.items():
            value, meta = self._extract_value(raw_entry)
            if value is None:
                continue

            if self._is_valid_array(value):
                arr = np.ascontiguousarray(value)
                datadef[field] = (arr.dtype.type, arr.ndim)
                field_values[field] = arr
            elif isinstance(value, str):
                datadef[field] = (str, 1)
                field_values[field] = value
            elif isinstance(value, (numbers.Number, np.generic)):
                scalar, dtype_type = self._coerce_scalar(value)
                datadef[field] = (dtype_type, 0)
                field_values[field] = scalar
            else:
                # Unsupported type (e.g. dict). Skip for now.
                continue

            if meta:
                meta_field = f"{field}__meta"
                datadef[meta_field] = (str, 1)
                field_values[meta_field] = json.dumps(meta, separators=(",", ":"))

        return datadef, field_values

    def _estimate_data_size(self, field_values: Dict[str, Any]) -> int:
        total = 0
        for value in field_values.values():
            if isinstance(value, np.ndarray):
                total += value.nbytes
            elif isinstance(value, str):
                total += len(value.encode())
            else:
                total += 16  # scalars
        return total

    def convert_to_buffer(
        self, calib_const: Dict[str, Dict[str, Any]]
    ) -> Tuple[memoryview, int, int]:
        """
        Convert calibration constants into a single buffer holding Configure+L1Accept.
        Returns (buffer_view, config_size, data_size).
        """
        debug = True
        if debug:
            t_start = time.perf_counter()
            t_build_start = t_start
        det_handles = {}
        det_field_values = {}

        # First pass: define Names and compute buffer sizes.
        total_bytes = 0
        field_count = 0
        config_edit = DgramEdit(
            transition_id=TransitionId.Configure, bufsize=64 * 1024 * 1024
        )

        for det_name, entries in calib_const.items():
            if not isinstance(entries, dict):
                continue
            datadef, field_values = self._build_datadef(det_name, entries)
            if not datadef:
                continue

            detid = self.det_info.get(det_name, det_name)
            detdef = DetectorDef(det_name, det_name, detid)
            det_obj = config_edit.Detector(detdef, self.alg, datadef)
            det_handles[det_name] = det_obj
            det_field_values[det_name] = field_values
            total_bytes += self._estimate_data_size(field_values)
            field_count += len(field_values)

        if not det_handles:
            raise ValueError("No calibration entries to convert.")

        if debug:
            t_build_end = time.perf_counter()
            t_blob_alloc_start = time.perf_counter()
        config_bufsize = 8 * 1024 * 1024
        data_bufsize = max(total_bytes + 1024 * 1024, 16 * 1024 * 1024)
        blob = bytearray(config_bufsize + int(data_bufsize))
        if debug:
            t_blob_alloc_end = time.perf_counter()
        view = memoryview(blob)
        config_view = view[:config_bufsize]

        if debug:
            t_cfg_save_start = time.perf_counter()
        config_edit.save(config_view)
        if debug:
            t_cfg_save_end = time.perf_counter()
        config_size = config_edit.size
        data_view = view[config_size : config_size + data_bufsize]

        if debug:
            t_data_edit_start = time.perf_counter()
        data_edit = DgramEdit(
            config_dgramedit=config_edit,
            transition_id=TransitionId.L1Accept,
            ts=0,
            bufsize=int(data_bufsize),
        )
        if debug:
            t_data_edit_end = time.perf_counter()

        if debug:
            t_add_start = time.perf_counter()
        for det_name, det_obj in det_handles.items():
            alg_container = getattr(det_obj, self.alg.name)
            field_values = det_field_values[det_name]
            for field, value in field_values.items():
                setattr(alg_container, field, value)
            data_edit.adddata(alg_container)
        if debug:
            t_add_end = time.perf_counter()

        if debug:
            t_data_save_start = time.perf_counter()
        data_edit.save(data_view)
        if debug:
            t_data_save_end = time.perf_counter()
        data_size = data_edit.size

        if debug:
            t_end = time.perf_counter()

        used_size = config_size + data_size
        return view[:used_size], config_size, data_size


def _extract_alg_fields(alg_obj) -> Dict[str, Any]:
    """
    Convert the attributes exposed by a DgramEdit algorithm entry into a plain dict.
    Metadata fields (``__meta`` suffix) are re-attached to their base field as tuples.
    """
    values: Dict[str, Any] = {}
    meta_fields: Dict[str, Any] = {}

    for attr in dir(alg_obj):
        if attr.startswith("_"):
            continue
        val = getattr(alg_obj, attr)
        if callable(val):
            continue
        if attr.endswith("__meta"):
            base = attr[: -6]
            if isinstance(val, (bytes, bytearray)):
                val = val.decode()
            if isinstance(val, str):
                try:
                    meta_fields[base] = json.loads(val)
                except json.JSONDecodeError:
                    meta_fields[base] = val
            else:
                meta_fields[base] = val
        else:
            values[attr] = val

    for field, val in list(values.items()):
        if field in meta_fields:
            values[field] = (val, meta_fields[field])
    return values


def load_calib_xtc(path: str = "/dev/shm/calib.xtc2") -> Tuple[Dict[str, Dict[str, Any]], Any]:
    """
    Load calibration constants from an xtc2 file produced by CalibXtcConverter.

    Returns:
        tuple(dict, bytearray): Calibration dict compatible with dsparms.calibconst and the
        backing buffer keeping NumPy views alive.
    """
    blob = bytearray(Path(path).read_bytes())
    return load_calib_xtc_from_buffer(blob)


def load_calib_xtc_from_buffer(buffer: Any) -> Tuple[Dict[str, Dict[str, Any]], Any]:
    """
    Interpret an in-memory xtc2 buffer and return calibration constants plus the owner object.

    Args:
        buffer: bytes-like object containing Configure+L1Accept dgrams
    """
    mv, owner = _as_memoryview(buffer)
    config = dgram.Dgram(view=mv, offset=0)
    payload = dgram.Dgram(config=config, view=mv, offset=config._size)

    calib_const: Dict[str, Dict[str, Any]] = {}
    det_names = [name for name in dir(config.software) if not name.startswith("_")]
    for det_name in det_names:
        if not hasattr(payload, det_name):
            continue
        segments = getattr(payload, det_name)
        if not segments:
            continue
        segment = segments[0]
        det_values: Dict[str, Any] = {}
        for alg_name in dir(segment):
            if alg_name.startswith("_"):
                continue
            alg_obj = getattr(segment, alg_name)
            if callable(alg_obj):
                continue
            det_values.update(_extract_alg_fields(alg_obj))
        if det_values:
            calib_const[det_name] = det_values

    return calib_const, owner


def _as_memoryview(buffer: Any) -> Tuple[memoryview, Any]:
    """Return a byte-format memoryview and the object keeping the data alive."""
    if isinstance(buffer, memoryview):
        mv = buffer
        if mv.format != "B":
            mv = mv.cast("B")
        return mv, buffer
    try:
        mv = memoryview(buffer)
    except TypeError:
        tmp = bytearray(buffer)
        mv = memoryview(tmp)
        return mv, tmp
    if mv.format != "B":
        mv = mv.cast("B")
    return mv, buffer
