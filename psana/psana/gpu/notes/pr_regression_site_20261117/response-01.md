# Response to round 1

- SITE-001 accepted/fixed: rank observations precede placement requirements,
  including raw physical topology without SMT rejection, affinity/allowed mask,
  job/step and CPU/GPU visibility. GPU identity or query error is saved before
  expected UUID validation. Mask errors include actual/expected details. Offline
  mismatch test verifies the diagnostic survives. Adapter launch flags unchanged.
- SITE-002 accepted/fixed: draft schema is explicitly unaccepted. Published site
  links capture index path/hash; index covers validation. Document pinning both
  files, since native load_contract does not validate the extra capture link.
- SITE-003 accepted/documented: controller timeout is an advisory compatibility
  check, not preparation's own deadline. Conservative walltime includes serial
  scheduler query timeouts, cleanup and setup. Validation uses 90-second steps
  and 22-minute job limit. No timeout/signal qualification claim.
- SITE-004 accepted/fixed: batch CUDA visibility retained alongside job GPU IDs,
  full NVIDIA inventory and observed per-rank identities. No GPU fallback.
- SITE-005 accepted/fixed: tests cover non-batch/multinode contexts, missing
  allocated GPU, existing output/capture directory, runtime/source mutations,
  disappearing NIC and rank mismatch diagnostics. CPU no-CUDA is documented as
  an assertion from the actual in-rank import/mapping guard.

First expanded suite exposed a fixture variable shadowing error in its mutation
injection, not production; corrected and reran. Both test logs are retained.
Hardware evidence remains pending, including explicit --exact/mask_cpu and zero
GPU launch viability. Offline success does not settle those questions.
