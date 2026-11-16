# Responses to round 1

- SCALE-001 — fixed. Local scontrol confirms CR_CORE_MEMORY. srun now uses even
  logical CPU counts 20/12/8/6 for 6/10/14/18 ranks, requiring 60/60/56/54
  physical cores. Local srun help does not expose --whole, so that unsupported
  option was not added. The existing runtime intersection/probe gate requires
  all 54 physical cores with both siblings before GPU tests. Offline tests now
  assert evenness and tasks*ceil(c/2)<=64, rather than repeating the formula.
- SCALE-002 — accepted documented limitation, as offered by the reviewer.
  README explicitly states 33 callbacks / 16 BDs is 2.06 on average, not a
  minimum, and an idle BD safely aborts this bounded pilot without a retry.
  Coverage will not be relaxed. Larger witness cases require a scoped reviewed
  adjustment if this risk materializes.
- SCALE-003 — fixed. test_handoff.py creates complete actual-schedule pilot and
  screen attempts (15/21 numeric Slurm steps), raw hashes, original and
  intersected probes, native-test receipts, memory, ranges, and provenance.
  It exercises audit(), the positive screen validate_plan() path, report()
  with deliberately different per-block baselines, and runtime affinity()
  against synthetic sysfs. Corruptions cover missing steps, changed masks,
  incorrect NodeList/device, memory hashes, pilot hash, and swapped baseline.
  Only external hardware/data reads are mocked. All 16 tests pass in 9.593 s.
- SCALE-004 — fixed documentation. README names fixed single SMD0/EB helpers
  as possible serial bottlenecks and explicitly limits causal interpretation.
  Per-rank CPU-utilization instrumentation is deferred; throughput alone cannot
  attribute a plateau to helper CPU, storage, parsing, or GPU.

To resolve denied reads from round 1, evidence/native-unit.json now contains
  the full 1023-test receipt with its unchanged SHA256 in checks.json. The
  exact installed gpu_placement source and its verified build-manifest hash
  are also included under evidence/ for inspection inside the review checkout.
