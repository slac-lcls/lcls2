# Independent run51 input preparation review
Reviewed SHA df46718002367e56a7bca5c0d1d844342df2493c
Base6b87fad3f952a8fdf2e4014846177f362a274cb8

Repository slac-lcls/lcls2 is public (GitHub API verified private=false,
visibility=public). Review only committed source/docs and compact validation
under psana/psana/gpu/scripts/performance/regression and
psana/psana/gpu/notes/pr_regression_inputs_20261117. Do not read data files, large
scratch plans, credentials or unrelated user files. Context contains new source
for this public repository and non-sensitive sizes/hash/provenance metadata;
no detector payload, credentials or private organizational documents.

Read scope.md, source-scan.json, validation-01.json and prepare.sbatch. User
approved private input preparation; no copy job submitted yet. Review new
prepare_inputs.py, native_contract.py exact-range extensions, tests and docs.
XTC header source authority: xtcdata/xtcdata/xtc/Dgram.hh, Xtc.hh, TimeStamp.hh.
SMD offsets: psana/psana/psexp/event_manager.py. Dgram decoding is independent of
benchmark DataSource/Events iteration. Actual SMD scan passed with pinned3.9CPU
Dgram:17872available,10kselected,335571760000eventbytes,602108910258copybytes;
83 SlowUpdate gaps require84 disjoint ranges per stream, no gap bytes counted.
All five stream event/step/transition identities and32segments agree.

Coordinator evidence:55offline tests PASS (26adapter,20controller,9preparation),
bash syntax and diffcheck pass. Check reference/ranges correctness, hash/copy
integrity, headers, source invariants, failure/no-overwrite behavior, compute-only
copy and frozen launcher. Real large copy evidence is subsequent, not claimed.
No framework/profile/qualification changes. Legacy v1 supported. One premium
lcls_g compute allocation:1GPU reserved unused,8CPU,32GiB,30min. Exact layout
stripe1/1MiB, actualOSTs recorded. First direct extension-import experiment crashed;
normal psana/numpy import with MPICH GPU support disabled succeeded. Performance
runtime qualification separate. No nested review or commands; file-reading only.
