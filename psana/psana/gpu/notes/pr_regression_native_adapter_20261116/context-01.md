Review the implementation at 64acad1438955ee1170ec304f376cbb9aa22295b against base
5a76c182f7d56a86d4dba43f5baf749400366bcf. Do not delegate, execute, or modify.
This is user-requested implementation + independent Claude review, max3 rounds.

Scope and validation:
- psana/psana/gpu/notes/pr_regression_native_adapter_20261116/scope.md
- same directory validation-01.json (18 adapter,20 controller tests pass; catalog10/2082/275 pass)
- same directory status.json
- psana/psana/gpu/scripts/performance/regression/NATIVE.md

Inspect new native_contract.py, native_worker.py, perlmutter_adapter.py,
test_native.py, and controller campaign.py changes (timeout passed to adapter;
optional nested native artifact index validation). README and template doc links
are updated. All files are in psana/psana/gpu/scripts/performance/regression
except psana/psana/gpu/docs/performance/templates/pr-regression.md.

Acceptance for this scoped implementation: correct fail-closed native launch,
raw-workload reference/byte/identity checks, runtime/placement/cache evidence,
synchronous separate warmup/measured steps, cleanup bounded to own step,
separate diagnostics, tests of adapter failure paths (not duplicated psana unit
invariants), truthful docs, frozen archives/profile unchanged. Inspect source
for concrete API defects despite native execution being explicitly deferred.
Source API base is features/psana2-gpu199575205 (included in this checkout):
psana/psana/psexp/mpi_ds.py, run.py, events.py, node.py, ds_base.py;
psana/psana/gpu/gpu_events.py, gpu_kvikio_read.py, gpu_producer.py, gpu_task_batch.py.
The historical PR299 worker in scripts/pr299_perlmutter/harness can be read as
context only; not imported or modified. KvikIO26.10 defaults.get/set is required;
old PR299Python3.9 KvikIO API was inspected and is not claimed compatible.

Project guidance: actual native source wins over stale notes. Preserve MPI
SMD0/EB/BD topology, owner lifetimes/drains, no fake GDS or performance claims.
Counters must distinguish events, EB deliveries, GPU executions, physical reads;
never pretend unavailable visit counts or memory peaks are zero. cProfile host
call counts are separate diagnostic evidence, not clean timing. Do not infer
GPU acceptance from offline mocks. Raw GPU GpuTask uses dense jungfrau.raw input
preparation with no user kernel/output, not descriptor-only staging.

Qualification remains REQUIRED FUTURE WORK, not this implementation acceptance:
no input copy/reference generator/native manifests, no MPI/GPU smoke/job, no
real scheduler cleanup test, no paired results/overhead measurement, no component
bounds qualification. Default profile flags remain false. SDF launcher excluded.
Exact native builds/reference preparation and compute campaign need separate
qualification. No existing installation is asserted qualified by this change.
Do not demand hardware results as if they were claimed, but flag actual code or
contract defects and missing offline failure tests when necessary. A passing code
review must not imply qualified performance execution. Review findings must have
stable IDs and exact inspected evidence; record any inaccessible evidence.
