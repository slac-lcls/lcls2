# Review: unique 10k sample10 project-scratch fixture

Review the exact commit supplied by the helper. Base is
f1b8bb7e5265e7044f6a27289b65c9285f09c703. Read scope.md, README.md, status.json,
campaign.py, fixture.py, verify.py, test_fixture.py, run.sbatch, preflight.json,
validation-summary.json, job-43538554.log and access.txt in
notes/mfx_expand10_10k_20261106. Read the linked main proposal follow-up too.
The coordinator will add final compact evidence before invoking review.

Scope: user authorized the first stage of the 10k single-node campaign: generate
10,000 unique source events in prjdatpsana scratch and validate the fixture.
No GPU work, new public run, database writes, changed calibration, psana core
changes, artificial scan steps or repeated 400k dataset. Existing Stage 2 and
capacity evidence must remain unchanged. Independent review loop is explicitly
requested, maximum three rounds, coordinator retains implementation ownership.

Compare new fixture.py/verify.py/test_fixture.py with their Stage 2 originals.
Their intentional fork preserves previous artifact code-hash bindings. Writer
changes are 10k upper bound, directory mode/identity wording, progress messages.
Verifier changes are the bound, per-stream schema cache, keyed timestamp/panel
seed witnesses with duplicate rejection, progress messages and scope wording.
Per-pixel checks are retained; the DataSource lookup changes complexity only.
The shared xtc_fixture.py parser/timestamp helpers and source snapshot manifest
remain unchanged. No dynamic rewriting of historical modules is used.

Inspect exact size arithmetic, full source/header/schema validation, calibration
byte/shape/hash checks, project path and collision checks, permissions, bounded
window/timeline closure, phase/failure semantics, full stream readback, preserved
non-JF payloads and JF scalar fields, SMD references, native DataSource traversal,
event identities and all 32 panels. File existence alone must not imply accepted.
Project filesystem free space is not claimed as a quota. Check claims against
the final completed CPU job and per-file/readback evidence, not against the
preliminary planning estimate. Strictly distinguish CPU format/seed acceptance
from prior GPU correctness and future performance work.

Full generation.json and verification.json and the controlled job evidence are
copied inside this detached checkout under build/tenk-review-evidence/. Read
its copy-index.json for paths/hashes and compact summaries for navigation. The
full verification report is large: use targeted reads if needed. Original
scratch paths are provenance only and may be outside your read sandbox. The
coordinator independently computes hashes and source binding; you have read-only
tools and cannot claim to have executed hash commands, tests or inspected source
XTC binary contents. Report any missing/denied evidence as a limitation.

Do not delegate or run a recursive review loop. Return the exact reviewed SHA,
verdict, limitations and stable finding IDs beginning TENK-001. Distinguish
blocking acceptance failures from optional hardening outside this fixed scope.
