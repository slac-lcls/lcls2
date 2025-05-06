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
