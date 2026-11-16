# Why one CUDA stream has no pressure drains

Investigate the observed 0 versus 142 steady-window drains with one versus two
CUDA streams. Identify the actual exception reason, input ownership/credit
state, available reader slots and framework budget at rejection. Distinguish
byte-budget exhaustion from stream-credit backpressure, both of which use
GpuMemoryPressureError in the frozen runtime.

Add a bounded, process-local diagnostic observer and configurable framework
budget/coalescing target to the existing performance runner. Keep defaults and
all GPU kernels, psana core, fixtures and constants unchanged. Run four focused
1k A100 cases on one node: one/two streams at 4 GiB and 8 MiB target; two streams
at 8 GiB and 8 MiB target; two streams at 4 GiB and 512 KiB target. These are
causal diagnostics, not a 10k scaling or sustained-performance campaign.

Acceptance: observer tests for failure cause, snapshot ownership and patch
restoration; existing preparation host tests; all native event/bin checks;
runtime/source hashes; independent restricted Claude review of implementation
and conclusions, with response/re-review as needed (three-round limit).
Save evidence in project scratch and compact findings in this branch.
