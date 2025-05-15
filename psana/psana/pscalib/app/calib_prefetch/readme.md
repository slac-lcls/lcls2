# calib_prefetch

The `calib_prefetch` module provides a utility for pre-fetching and caching detector calibration constants for LCLS experiments. It enables efficient reuse of calibration data by storing them in a shared-memory-friendly format (e.g. `/dev/shm`) to minimize redundant fetches and startup time for high-performance data sources.

## Features

- Automatically detects the latest usable run from XTC files.
- Fetches calibration constants per detector from the LCLS calibration database.
- Stores calibration constants in `.pkl` files with atomic `.inprogress` write protection.
- Reuses existing calibration data when possible (based on detector UID match).
- Can be run as a background CLI tool for continuous prefetching.
- Easily integratable with `ShmemDataSource` and `DataSourceBase`.

## CLI Usage

```bash
python -m psana.pscalib.app.calib_prefetch \
  -e xpptut15 \
  --xtc-dir /sdf/data/lcls/ds/xpp/xpptut15/xtc \
  --output-dir /dev/shm \
  --interval 5 \
  --log-level DEBUG \
  --timestamp
```

### Options

| Option           | Description                                         |
|------------------|-----------------------------------------------------|
| `-e`, `--expcode` | Experiment code (e.g. `xpptut15`)                  |
| `--xtc-dir`       | Path to the experiment’s XTC directory             |
| `--output-dir`    | Directory to store cached calibration files        |
| `--interval`      | Time in minutes between calibration checks         |
| `--log-level`     | Logging level (`DEBUG`, `INFO`, etc.)              |
| `--timestamp`     | Whether to include timestamps in log messages      |
| `--detectors`     | List of detector names to precompute               |

## Output Format

The calibration file is written as:

```
calibconst_{expcode}_r{runnum:04d}.pkl
```

It contains:
```python
{
  "det_info": {det_name: unique_id, ...},
  "calib_const": {det_name: calib_dict, ...}
}
```

## Integration Example (Python)

You can use `ensure_valid_calibconst()` in your own code:

```python
from psana.pscalib.app.calib_prefetch import calib_utils

calib_const, runnum = calib_utils.ensure_valid_calibconst(
    expcode="xpptut15",
    latest_run=123,
    latest_info=detector_info,
    xtc_dir="/path/to/xtc",
    skip_calib_load=["det0"]
)
```


## 🧠 Detector Caching Support

In addition to calibration constant prefetching, `calib_prefetch` can now also trigger detector-side caching of geometry-related attributes to reduce initialization time during event processing.

### 🔧 New Options

| Option                | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `--shmem`             | Run in shared memory mode (typically required for triggering detectors)    |
| `--detectors`         | List of detector names to precompute and cache `_calibc_` attributes        |
| `--check-before-update` | Skip updating if detector UID matches previously cached state            |

When `--detectors` is provided, the tool will call `det.raw.image(evt)` internally on one event to ensure expensive geometry arrays are computed and saved using `DetectorCacheManager`.

These arrays (e.g. pixel coordinates, masks, and mappings) are then saved in a structured pickle file in the specified output directory (default: `/dev/shm`).

### 📁 Detector Cache Format

Each detector's cache is saved under:

```
/dev/shm/<detector_name>_calibc_cache.pkl
```

Format:
```python
{
  "raw": {
    "_pix_rc": np.ndarray,
    ...
  },
  "fex": {
    ...
  }
}
```

These are loaded later by the `Detector` interface (e.g. `det.raw`) during `DataSource` setup if `use_calib_cache=True`.

### 🧪 Example Usage

```bash
python -m psana.pscalib.app.calib_prefetch \
  --expcode mfx101332224 \
  --xtc-dir /sdf/data/lcls/ds/mfx/mfx101332224/xtc \
  --output-dir /dev/shm \
  --shmem shmem_test_12345 \
  --detectors jungfrau \
  --log-level DEBUG
```

This will write both the calibration constants (`calibconst.pkl`) and detector-side `_calibc_` geometry caches for the `jungfrau` detector to `/dev/shm`.

### 🧩 Integration Notes

- `calib_prefetch` should be launched at DAQ start by a control process such as `daqmgr`.
- Use `use_calib_cache=True` and `cached_detectors=["<detector>"]` when creating the downstream `DataSource`.
- Detector-side loading will automatically populate `_calibc_` attributes and skip redundant computation in `image(evt)` or `geometry()`.

## Developers

The logic is modularized in `calib_utils.py`, supporting:

- `get_latest_run()`
- `load_existing_run()`
- `needs_update()`
- `update_calib()`
- `ensure_valid_calibconst()`

These can be reused inside the DAQ pipeline, including:
- `psana.psexp.ds_base`
- `ShmemDataSource`
- `drp_ds.py`

## License

SLAC/LCLS internal use.
