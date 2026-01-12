import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from psana import dgram
from psana.psexp.calib_xtc import (
    CalibXtcConverter,
    load_calib_pickle,
    load_calib_xtc,
    load_calib_xtc_from_buffer,
)

try:
    import zstandard as zstd
except ImportError:
    zstd = None


CALIB_PICKLE = Path(os.environ.get("PSANA_CALIB_PICKLE", "/dev/shm/calibconst.pkl"))


def _unwrap_entry(entry):
    """
    Mirror CalibXtcConverter tuple handling for the pytest assertions.
    """
    meta = None
    value = entry
    if isinstance(entry, (tuple, list)) and entry:
        value = entry[0]
        if len(entry) > 1 and isinstance(entry[1], dict):
            meta = entry[1]
    return value, meta


def _env_flag(name, default="0"):
    value = os.environ.get(name, default).strip().lower()
    return value in ("1", "true", "yes", "on")


def _size_mib(size_bytes):
    return size_bytes / (1024 * 1024)


def _get_zstd_level():
    level_raw = os.environ.get("PSANA_CALIB_COMPRESS_LEVEL", "").strip()
    if not level_raw:
        return 3
    try:
        return int(level_raw)
    except ValueError:
        return 3


@pytest.mark.skipif(not CALIB_PICKLE.exists(), reason="calibration pickle not present")
def test_calib_pickle_roundtrip(tmp_path):
    """
    Convert calibconst.pkl into xtc2 and ensure the resulting payload can be read
    back via psana.dgram without copying array contents.
    """
    load_start = time.perf_counter()
    container = load_calib_pickle(str(CALIB_PICKLE))
    load_duration = time.perf_counter() - load_start
    print(f"Loaded calibration pickle in {load_duration:.3f} s")
    if container is None:
        pytest.skip("calibration pickle missing")

    calib_dict = container.get("calib_const") or container
    converter = CalibXtcConverter()
    out_path = tmp_path / "calib.xtc2"
    convert_start = time.perf_counter()
    buffer_view, config_size, data_size = converter.convert_to_buffer(calib_dict)
    with open(out_path, "wb") as fd:
        fd.write(buffer_view[:config_size])
        fd.write(buffer_view[config_size : config_size + data_size])
    convert_duration = time.perf_counter() - convert_start
    print(f"Converted calib pickle to xtc2 in {convert_duration:.3f} s")

    # Locate a detector with pedestals so we can do a byte-for-byte comparison.
    target_det = None
    pedestals = None
    ped_meta = None
    for det_name, det_fields in calib_dict.items():
        if not isinstance(det_fields, dict):
            continue
        if "pedestals" not in det_fields:
            continue
        field_val, field_meta = _unwrap_entry(det_fields["pedestals"])
        if isinstance(field_val, np.ndarray):
            target_det = det_name
            pedestals = np.ascontiguousarray(field_val)
            ped_meta = field_meta
            break

    if target_det is None or pedestals is None:
        pytest.skip("no detector with pedestals found in calibration pickle")

    disk_bytes = out_path.read_bytes()
    if _env_flag("PSANA_CALIB_COMPRESS"):
        if zstd is None:
            pytest.skip("zstandard module not available for compression test")
        original_size = len(disk_bytes)
        level = _get_zstd_level()
        compress_start = time.perf_counter()
        compressor = zstd.ZstdCompressor(level=level)
        compressed = compressor.compress(disk_bytes)
        compress_time = time.perf_counter() - compress_start
        compressed_size = len(compressed)
        print(
            "Compressed calib.xtc2 %.2f MiB -> %.2f MiB in %.3f s (zstd level %d)"
            % (_size_mib(original_size), _size_mib(compressed_size), compress_time, level)
        )

        decompress_start = time.perf_counter()
        decompressor = zstd.ZstdDecompressor()
        decompressed = decompressor.decompress(compressed)
        decompress_time = time.perf_counter() - decompress_start
        print(
            "Decompressed calib.xtc2 %.2f MiB -> %.2f MiB in %.3f s"
            % (_size_mib(compressed_size), _size_mib(len(decompressed)), decompress_time)
        )
        assert decompressed == disk_bytes
        disk_bytes = decompressed
    cfg_len = config_size
    config_view = memoryview(disk_bytes)[:cfg_len]
    data_view = memoryview(disk_bytes)[cfg_len : cfg_len + data_size]

    config_dgram = dgram.Dgram(view=config_view, offset=0)
    calib_dgram = dgram.Dgram(config=config_dgram, view=data_view, offset=0)

    assert hasattr(calib_dgram, target_det)
    det_container = getattr(calib_dgram, target_det)
    assert det_container, "detector has at least one segment"
    calib_alg = getattr(det_container[0], "calib")

    assert hasattr(calib_alg, "pedestals")
    np.testing.assert_allclose(calib_alg.pedestals, pedestals)

    if ped_meta:
        assert hasattr(calib_alg, "pedestals__meta")
        loaded_meta = json.loads(calib_alg.pedestals__meta)
        assert loaded_meta == ped_meta

    # Validate the helper that rehydrates calibconst dictionaries from xtc2.
    converted, backing = load_calib_xtc(str(out_path))
    assert backing  # keep buffer alive
    assert target_det in converted
    loaded_entry = converted[target_det]["pedestals"]
    if ped_meta:
        loaded_array, loaded_meta = loaded_entry
        assert loaded_meta == ped_meta
    else:
        loaded_array = loaded_entry
    np.testing.assert_allclose(loaded_array, pedestals)

    converted_buf, backing_buf = load_calib_xtc_from_buffer(memoryview(disk_bytes))
    assert target_det in converted_buf
    buffer_entry = converted_buf[target_det]["pedestals"]
    if ped_meta:
        buf_array, buf_meta = buffer_entry
        assert buf_meta == ped_meta
    else:
        buf_array = buffer_entry
    np.testing.assert_allclose(buf_array, pedestals)
    assert backing_buf is not None
