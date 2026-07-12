# psana GPU tests

The automated suite intentionally has two layers:

- `unit/test_core.py` contains fast, CPU-only invariants for metadata order,
  reusable-buffer lifetime, MPI batch unpacking, GPU-rank selection, fatal I/O
  handling, and default result routing.
- `integration/test_pixel_exact.py` is the single GPU/data acceptance test. It
  compares the integrated `DataSource(gpu_det=...)` result with normal psana
  calibration for the same timestamps and covers single-event and batched slot
  reuse.

Run them with:

```bash
python -m pytest -q psana/psana/tests/gpu/unit
python -m pytest -q -m slow psana/psana/tests/gpu/integration/test_pixel_exact.py
```

The acceptance dataset defaults to public Lysozyme Jungfrau data,
`mfx100848724` run 51. The older run-77 smoke tests were removed because their
effective mask makes calibrated output entirely zero, allowing incorrect
pixel/segment mappings to pass.

Performance measurements are scripts, not pytest assertions. Multi-rank GPU
validation is also explicit because it requires a Slurm/MPI allocation:

```bash
bash psana/psana/gpu/scripts/run_multi_gpu_test.sh
```

Do not add another external-data smoke test for shape, dtype, event count, or
NaNs alone; those properties are already implied by pixel-exact equality. Add
a test only for an independent failure mode that the two existing layers do
not cover.
