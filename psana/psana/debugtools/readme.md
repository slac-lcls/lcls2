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

For questions or contributions, contact the LCLS DAQ/Data Systems team.
