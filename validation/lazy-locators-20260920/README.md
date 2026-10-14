# On-demand configured locator views

The parser still decodes all configured fields with the existing initialization
and stream-grouped location kernels. `_locate_configured()` now retains only the
combined backing, configured handle indices, and shared ready event. It creates
no per-handle Python wrappers.

`locate(handle)` validates the handle, checks its wrapper cache, then checks the
configured index mapping. A configured cache miss creates and caches a contiguous
view of `backing[index, :n_dgrams]`, sharing the configured-ready event. It submits
no CUDA work and allocates no device storage. Unconfigured handles retain their
single-field decoder and allocator. Canonical gathering uses combined storage
directly, leaving the wrapper cache empty.

## Isolated validation build

Copied `validation/pre-bulk-review-20260918/install` to ignored `install/`, then
overlaid the current `gpudgram/parser.py`. All GPU Python modules were compared
byte-for-byte with current sources. `sources.json` records source and modified
integration-test SHA256 hashes. The CUDA `_kernel_source` AST is identical to
the preserved `install_psana` benchmark build. No recorded benchmark install was
updated, and no new wall-time speedup is claimed.

```bash
source setup_env.sh
source install_psana/activate.sh
export PYTHONPATH="$PWD/validation/lazy-locators-20260920/install/lib/python3.9/site-packages"
python -m pytest -q psana/psana/tests/gpu/unit -m 'not gpu'
sbatch validation/lazy-locators-20260920/correctness.sbatch
```

CPU validation: **168 passed**, recorded in `unit.log`.
The combined unit and GPU integration suite: **206 passed, 6 slow tests
deselected**, recorded in `correctness-38676735.log`. Job **38676735** on
**sdfampere027** completed with exit **0:0** (24 seconds allocation elapsed).
Hardware: A100-SXM4-40GB, UUID `GPU-d4eeaffa-a368-71d7-b83b-9315de587942`;
CuPy 13.6.0, CUDA runtime 12090, driver 575.57.08. Logs include the existing
Kerberos-credentials and pytest-asyncio configuration warnings; no tests failed.
`git diff --check` passed. These are correctness runs, not performance samples.

## Assertions added

- Parsing constructs zero wrappers, including empty inputs.
- First access constructs one wrapper; repeated access returns that same object.
- Configured access bypasses the decoder/CuPy scheduling path and allocator.
- Output index zero and nonzero indices address the original backing, using its
  capacity stride correctly through batch sizes 5, 2, 0, and 7.
- Requested views share the explicit ready event, including cross-stream use;
  unused handles have no wrappers and pool memory accounting is unchanged.
- Existing single-field reference comparisons cover missing fields, invalid
  inputs, duplicate blocks, and the unconfigured-handle fallback.
- Canonical gathering creates zero cached wrappers for same/cross-stream,
  uint16/float32, tail/empty/growing inputs, and matches both independently
  generated pixels and the previous per-field gather path.

## Bulk integration

The bulk branch's `InputWindow` must explicitly retain
`batch.configured_locations().ready`, independently of `_locators`. Deduplicate
that shared event while retaining separate unconfigured-field and consumer
events. Bulk-read code is not changed in this follow-up.
