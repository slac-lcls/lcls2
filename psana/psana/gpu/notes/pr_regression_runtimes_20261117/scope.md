# Reusable runtime preparation

User authorized step1 of the reusable campaign workflow: build isolated exact
revisions, verify dependencies, produce runtime manifests; use Claude review.
Branch results/psana2-gpu-scaling, base66da2d9894a8b5486c767675454f52cac933eb65.
Maximum three independent review rounds. Preserve prior studies and input data.

Implement prepare_runtime.py build/capture/check-pair commands, documentation and
focused failure/identity tests. Build uses a new detached source worktree and
new Meson install/build directories, exact full SHA, explicit Python and shared
six-key environment. No destructive clean/resume, source changes or environment
package installation. Capture requires an allocated Perlmutter GPU node and
runs small import/MPI/CUDA/NVRTC/KvikIO probes, never experiment data or a campaign.
Record source/build/install/tool/environment/dependency/library identities;
completion receipts last, failures retained. Manifest generation is not native
adapter/performance qualification or functional acceptance of a PR.

Scoped adapter integration: shared dependency environment must coexist with
separate install paths. New manifests declare isolated-install-v1; launch prepends
only the selected install paths. Legacy manifests retain existing behavior.
New manifests also pin installed support libraries/package files. Protect against
mixed imports and altered files. No default profile qualifications enabled.

Validate offline failure paths and real build/capture where available. Use the
known API revision199575205 as a driver validation fixture, not an implicit future
PR baseline. Any compute validation is bounded runtime probing only, premium
lcls_g with normal allocation checks. Review implementation before compute probe;
review actual evidence afterward. Commit/push reviewed result to existing branch.
