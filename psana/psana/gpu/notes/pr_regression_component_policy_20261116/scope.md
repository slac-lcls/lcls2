# Scope: performance boundaries without duplicate correctness tests

User directs the PR performance template to protect major component performance,
especially batch scheduling versus repeated per-event/item host work, while
avoiding duplication of unit tests. Base ce1b9cdba323574efb76bf7dac2d7660e9e30ef5.
Branch results/psana2-gpu-scaling; apply existing implementation/review loop,
maximum three rounds. Documentation/agent-contract change only: map existing
tests, prescribe native counters and qualified component boundaries, distinguish
clean timing from diagnostic instrumentation and manual campaign acceptance
from implemented controller gates. Keep six cases, n_gpu_streams=1, existing
profile/controller/tests, frozen historical harnesses and runtime unchanged.
No new native adapter, timing/counter gate implementation, benchmark, invented
numeric counter bound, or rerun of unrelated unit/device suites is claimed.
Checks: cited source/test functions, local links, catalog and whitespace; independent
Claude review at exact SHA. No new tests for this documentation change.
