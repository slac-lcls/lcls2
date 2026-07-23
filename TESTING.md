# Testing psana Pip Wheels

## Available Wheels

```
psana-4.3-cp311-cp311-linux_x86_64.whl  (13 MB) - Python 3.11
psana-4.3-cp313-cp313-linux_x86_64.whl  (13 MB) - Python 3.13
```

## Quick Test

### Python 3.11:
```bash
conda create -n test-psana python=3.11 numpy h5py -y
conda activate test-psana
pip install psana-4.3-cp311-cp311-linux_x86_64.whl
python -c "from psana import DataSource; print('✓ Works!')"
```

### Python 3.13:
```bash
conda create -n test-psana python=3.13 numpy h5py -y
conda activate test-psana
pip install psana-4.3-cp313-cp313-linux_x86_64.whl
python -c "from psana import DataSource; print('✓ Works!')"
```

## Full Test Suite

After installation, run these tests:

### Test 1: Basic Imports
```python
python << 'EOF'
from psana import DataSource, dgram, container
from psalg.utils import syslog
import psana.peakFinder_ext
import psana.shmem
print("✓ All imports successful")
EOF
```

### Test 2: No Version Mixing
```python
python << 'EOF'
import sys
wrong = [p for p in sys.path if 'python3.9' in p or 'python3.10' in p]
assert not wrong, f"Wrong Python in path: {wrong}"
print("✓ No version mixing")
EOF
```

### Test 3: Extensions Load
```python
python << 'EOF'
import psana.dgram
print(f"✓ dgram loaded from: {psana.dgram.__file__}")
EOF
```

### Test 4: Console Scripts
```bash
which detnames && echo "✓ Scripts in PATH"
detnames --help > /dev/null && echo "✓ detnames works"
```

### Test 5: Executables
```bash
which xtcreader && echo "✓ xtcreader in PATH"
xtcreader --help 2>&1 | head -3
```

## What to Report

If testing succeeds:
- ✅ "All tests pass"
- Python version used
- Operating system

If testing fails:
- ❌ Which test failed
- Full error message
- Python version (`python --version`)
- OS (`cat /etc/os-release | grep PRETTY_NAME`)

## Known Warnings (Safe to Ignore)

You may see:
- `SyntaxWarning: invalid escape sequence` - Pre-existing in source
- Compiler warnings during import - Don't affect functionality

## Important

- ❌ Do NOT source `setup_env.sh`
- ❌ Do NOT set `PYTHONPATH` manually
- ✅ Use the wheel matching your Python version
