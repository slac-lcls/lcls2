# psana.debugtools

A growing collection of runtime debugging tools to support LCLS2 experiment operations.

## 📦 Tools

### `elog_error_scanner.py`

Scans Slurm log files for runtime errors during experiment runs and reports them alongside the associated experiment and run information.

#### 🔧 Features:
- Detects and filters only the base `Run*.h5` files in `smalldata/`, skipping part files
- Automatically determines the scratch location of each relevant user (e.g. `/sdf/scratch/users/e/espov`)
- Matches Slurm output timestamps to run file timestamps within a small time window
- Extracts and prints relevant error lines
- Identifies the associated experiment and run from the Slurm log

#### 🚀 Usage

```bash
python -m psana.debugtools.elog_error_scanner <experiment_name>
```

#### 🔍 Example

```bash
python -m psana.debugtools.elog_error_scanner rixl1032923
```

Output:
```
==> /sdf/scratch/users/e/espov/slurm-3093333.out (modified: 2024-04-03 14:27)  [exp: rixl1032923, run: 22]
    Line 42: ERROR: Unable to initialize detector
    Line 103: Exception: segmentation fault
```

#### 🧠 Error Patterns
The scanner looks for lines containing:
- `error`
- `exception`
- `traceback`

(case-insensitive)

#### 📁 Internals
- Experiment data is parsed from:
  `/sdf/data/lcls/ds/<instrument>/<experiment>/hdf5/smalldata`
- Scratch logs are scanned from:
  `/sdf/scratch/users/<first-letter>/<username>/slurm-*.out`

---

### Coming Soon:
- Markdown or CSV summary output
- ELog entry generation
- Support for live vs batch distinction
- Per-run error grouping

---

### simulate_smalldata_write.py

Simulates concurrent writing of `.smd.xtc2` files using real `xtc2` data and the `smdwriter` tool.

#### What it does:
- Detects stream files from a real experiment and run (e.g. `s000`, `s001`, ...)
- Creates symlinks in the given `output_path` to those files
- Launches one `smdwriter` process per stream in parallel
- Writes `.smd.xtc2.inprogress` files into `output_path/smalldata/`, and renames them to `.smd.xtc2` when done

#### Usage:
```bash
python -m psana.debugtools.simulate_smalldata_write <exp> <run> <output_path> [-m SECONDS] [-n EVENTS]
```

#### Example:
```bash
python -m psana.debugtools.simulate_smalldata_write rixl1032923 22 ~/tmp/debugtest -m 5 -n 100
```

#### Parameters:
- `exp`: Experiment name (e.g. `rixl1032923`)
- `run`: Run number (e.g. `22`)
- `output_path`: Where symlinks and smalldata are written
- `-m`: Delay in seconds during `smdwriter` (default: `10`)
- `-n`: Number of events to simulate (default: `200`)

Use this to simulate real-time smalldata writing for testing tools like live monitors or file watchers.

### scan_daq_logs_by_run.py

Scans DAQ log files by run timestamp, grouping all logs from the same invocation and reporting only the relevant errors and warnings in a concise, deduped format.

### Key features:

- Glob & Sort: Supply shell patterns (e.g., ~/2025/04/24*) to select log files.
- Group by Run: Files are grouped by their date_time prefix (e.g., 24_10:26:29).
- Error Detection: Matches case-insensitive error, segmentation fault, and any !! warnings.
- Ignore Filters: Skips slurmstepd: error noise lines.
- Consecutive Deduplication: Collapses only consecutive repeats of the same normalized error; resets suppression when a different line appears.

### Usage:
```bash
python -m psana.debugtools.scan_daq_logs_by_run <log_glob1> [<log_glob2> ...]
```
### Example:
```bash
python -m psana.debugtools.scan_daq_logs_by_run ~/2025/04/24*
```

For questions or contributions, contact the LCLS DAQ/Data Systems team.

