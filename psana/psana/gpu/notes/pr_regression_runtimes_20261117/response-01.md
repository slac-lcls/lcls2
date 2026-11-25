# Round1 response

RUN-001: launcher records scontrol allocation, Slurm variables and GPU list
before srun. EXIT trap records accounting after success or failure. A final
coordinator sacct read after job exit is still required because the batch job
can be RUNNING during its own trap. Probe records GPU UUID/model/ordinal, step
and CUDA visibility. Requested4CPU/1GPU/16G; actual full-node allocation recorded.
RUN-002: CPU guard rejects all requested GPU module namespaces and CUDA/NVRTC/
cuFile mappings before GPU imports. Tests exercise each form of indirect loading.
RUN-003: child environment key names and digest recorded, no arbitrary values.
Inherited environment still requires site qualification; legacy adapter unchanged.
RUN-004: documented exact limited dependency closure in RUNTIME and manifest.
RUN-005: every operation after capture mkdir now in try; early errors retain
failed.json. Parent rechecks build and child commit/install/receipt association.
Publication fsyncs a temporary inode and uses exclusive hard link rather than
rename (which could overwrite an existing output). Partial writes never publish.
RUN-006: different commits require disjoint nonnested installs, correctly rooted
package/pythonpath and distinct source worktrees. A/A requires same exact manifest.
Added mixed-policy, mixed-psalg, external-native-library and nested-prefix tests.
RUN-007: launcher embeds driver/helper SHA256 values. Before submission verify
both frozen blobs and launcher against reviewed commit, record that verification.

70 tests PASS; bash syntax/diff check PASS. No GPU capture submitted yet. The
real build receipt remains from initial driver50eabba86; build code is unchanged
by these capture/verification fixes, and capture rechecks source/install hashes.
