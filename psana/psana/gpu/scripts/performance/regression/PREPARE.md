# Prepare real run-51 inputs

`prepare_inputs.py scan --source /source/xtc --output /new/plan.json`
reads the five small SMD files with a pinned CPU psana Dgram decoder. It does
not iterate DataSource, Run, Events or any benchmark workload. Select an explicit
Python/install, set `MPICH_GPU_SUPPORT_ENABLED=0`, `PS_PARALLEL=none`, and suppress
bytecode. This preparation decoder need not be the future performance runtime;
its library hash and Python identity are recorded in the plan.

The fixed plan covers 10,000 of 17,872 events, streams3/5/7/8/9, all32 Jungfrau
segments, and native BeginStep indices. It verifies BeginRun identity and that
all five streams agree on event/step/transition sequences. SMD framing is checked
before decoding; truncated records, extra chunks, overlapping offsets or a
changed source fail. The first10k prefix contains83 SlowUpdate gaps per stream:
84 disjoint L1 ranges. Gap bytes are excluded from the event reference, selected
range hash and bandwidth denominator. Cache residency counts only requested L1
bytes, even when L1 ranges share an OS page around a transition gap.

Run preparation on an allocated compute node:

```bash
python -B prepare_inputs.py prepare --plan /new/plan.json \
  --plan-sha256 SHA256 --destination /private/new-directory
```

The destination must not exist. The command copies entire selected XTC and SMD
files (about561GiB), using at most five concurrent stream copies; it never changes
or evicts source data. Each source is hashed while copying, then the destination
is reread and independently hashed. Every selected bigdata header is checked
against the SMD timestamp, service and length. Inter-range gaps must be24-byte
SlowUpdate records. Source metadata/SMD hashes are checked before and after.

The directory is owner-only, data files become read-only and have link count1.
Lustre layout is explicitly one stripe,1MiB stripe size, with actual OST layout
recorded. This controls the new copy's layout; it does not promise the same OST
as the source. No cache-state or performance acceptance is implied by copying.

Outputs include `event-reference.json`, `dataset-manifest.json` (v2), and
`preparation-complete.json` with source/copy hashes, layout and provenance.
A started marker without the completion receipt means incomplete inputs. Failed
copies remain for diagnosis; no automatic removal, overwrite or resume. Data and
large plans stay in task scratch, not Git. Freeze/hash the tool and native_contract
helper before launch; retain Slurm logs and successful scheduler accounting.

Use the resulting paths/hashes and event digest/byte total when preparing a new
campaign profile. No default qualification flag is enabled by this command;
review input evidence before enabling applicable input receipts. Runtime builds,
placement, cache residency and native/A-A qualification are subsequent steps.
