# Plan review closure

Round2 passed e1b818be7fb0af225d0bb47bfb71ae96cb072157 with no blocking findings,
all required reads successful and no permission denials. All7round1 suggestions
are resolved. Runtime remains unbuilt/unmeasured at this exact PR head.

Two low-priority implementation prerequisites accepted:
- PLAN-R2-01: the four1003-event tail correctness cases will use the count2
  immutable view,at1/4BDs and1/2streams. They count against the6nodehourcap,
  as do all other native correctness,binding,sizing and pilot work.
- PLAN-R2-02: the current static inventory intentionally lists only the two
  changed unit regression files and six native suites. Before timed pilots,
  implementation review must enumerate/hash ALL files in the pinned PR GPU unit
  directory and save actual collected node IDs/counts,including parametrization.
  It must not treat the two unit files as the full suite.

These clarify implementation prerequisites without changing the reviewed
72case design. No further plan review cycle needed. Building,implementation
review and native acceptance remain future work after presenting the list.
No build,data staging,cache eviction or Slurm submission occurred this turn.

Coordinator checks: PR head metadata matches fetched67c327a; source excerpt
byte-equals git object; all30payload file sizes/inodes/mtimes match staging;
24unique configurations×3blocks=72; payload total21.9907TiB over fullscreen;
6hour additional cap leaves3.5536hours in the earlier12hour envelope if no
other work is charged. Native test inventory names/hashes are from the PR
objects,not runtime collection. git diff --check passes.
