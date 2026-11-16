# Round2 disposition and pilot launch

Round2 passed aca91d45230b3aa18a69f2472e16eebf2f177375 with no blockers.
All required reads succeeded. R1/R2 blockers resolved; R3-R6,R9,R10 fixed;
R7 remains an explicit coordinator walltime/budget check; R8 optional early
abort deferred because errors invalidate trials and900s timeout bounds reads.

New low-priority findings:
- R11 deferred: exact native readahead limit values were not captured. Worker
  file descriptors retain default kernel advice; pilot probe RANDOM advice is
  on separate descriptors. Native kernel/Lustre versions and missing counter
  availability are recorded. Do not claim measured/fixed kernel limit values.
- R12 handled by coordinator readback: bundle.verify passed and provenance
  commit exactly equaled the reviewed SHA before submission. Results acceptance
  will record execution commit and binary/source hashes explicitly.
- R13 handled by coordinator exact NodeList token comparison during native
  acceptance, supplementing audit.py's substring check. No source changes made
  after this review.

Pilot61118878 submitted2026-11-11 at00:31:29UTC,started00:31:33UTC on nid002657.
lcls_g/gpu_premium,one exclusive GPU node,128logicalCPUs,4allocated GPUs unused,
00:30:00 limit. Fresh attempt reader-pilot01; explicit chdir/stdout/stderr paths.
All sources and production binary match reviewed build pins. Iris before submit:
personal96.43%,project61.85%remaining. No low-balance trigger. No sweep yet.
