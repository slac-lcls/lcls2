# GPU memory, input access, and completion

**Status:** Current after user-kernel Stage 1b, 2026-09-26.

The runtime reads and parses GPU-selected streams and exposes leased input
fields. Built-in calibration, geometry, calibrated output slots, fixed-pair
CUDA IPC, and image-shaped automatic D2H have been removed. The previous version
of this document is preserved at `1d484d43d`; its image-output settings and
performance observations do not describe this input-only runtime.

## Ownership and budgets

Framework-owned device memory includes Configure tables, reader buffers,
parser/locator tables, and any explicitly selected dense input preparation.
`DenseInputPreparer` budgets raw/presence buffers, canonical routing, and gather
maps. Its host row-map uploads use pinned storage reported separately.

Before I/O, admission reserves allocation growth, including the overlap of old
and replacement storage. Cached capacity remains charged after execution ends.
Backing allocation owners retain charges while array aliases survive trimming.
Returning credit to the ledger is distinct from CuPy releasing cached free
blocks to CUDA. No calibrated outputs, derived constants, or geometry are
reserved by this path.

The per-BD quota defaults to a share of device memory based on the number of BD
workers assigned to that GPU. Admission leaves a 10% margin. User allocations,
CUDA contexts, KvikIO resources, and allocator overhead are outside the ledger;
the margin does not enforce a quota on arbitrary user allocations. Multi-EB
node-wide GPU assignment/accounting remains a [known issue](known_issues.md).

Communication batch size, input group size, execution subbatch size, and pool
depth are separate controls. Byte pressure can split a communication batch into
smaller executions or reduce overlap. An indivisible event that cannot fit fails
before I/O. `gpu_bulk_target_bytes` controls small-input grouping, not output
shape, event frequency, or a user kernel's allocation policy.

## Input windows and executions

`InputWindow` owns reader/parser backing. It can serve several executions;
planned uses, public input views, and registered CUDA consumers delay reuse.
`EventPool` owns execution streams and holds input leases while each subbatch
is active. Optional prepared arrays borrow reusable dense input slots; their
storage is retained until the execution's consumers finish.

```text
read completion -> GPU parse/locate -> optional dense prepare -> ready event
                                                              |
                    consumer stream waits -> work -> done event
                                                              |
                 retirement joins completion -> storage reusable
```

The reader waits for KvikIO futures before GPU parsing. This is not a claim
that file I/O itself is CUDA-stream ordered. Batched parser and gather launches
remain separate from physical read grouping.

A normal delivery sequence is:

1. Synchronize the outgoing execution's producer.
2. Yield its event envelopes with input leases still valid.
3. Allow field access to retain input owners and register consumer completion.
4. Retire execution leases; input windows remain live if other consumers need them.
5. Reuse only storage whose owners and completion tokens permit it.

Advancing the generator does not establish CUDA completion. Open input views can
outlive an execution and produce admission pressure. Owner references keep
allocations alive; CUDA events establish readiness. Both are required.

## Public input access

```python
field = evt.gpu.detector("jungfrau").field("raw", "raw")
raw_by_segment = field.on_cpu  # explicit host copy, cached independently
```

`field.on_gpu` returns independent device copies. For borrowed device views:

```python
with field.on_gpu_view(stream) as by_segment:
    # Queue input consumers on stream. Do not mutate borrowed input storage.
    consume(by_segment, stream)
```

The context retains all relevant input windows, waits for producer readiness,
and registers consumer completion. Multiple consumer streams are supported.
Do not use borrowed ndarray aliases after the context ends; retaining a Python
view alone does not prevent the framework from reusing its contents. Independent
copies and cached host values can be retained after event iteration advances.

There are no synthetic result keys for `.calib`, `.raw`, or `.image`.
`gpu_d2h_chunk_size` accepts only its retired zero default; nonzero requests
raise an error. No automatic output copy occurs. Explicit parsed-field access
can still copy input data to the host.

## Transitions, failures, and cleanup

BeginStep and EndRun drain dependent GPU input work before host transition
handling. No GPU calibration refresh runs. EndRun, exhausted input, early iterator
close, and `max_events` preserve the existing flush/close ordering.

Partial submission failures drain queued work before releasing owners. If CUDA
completion cannot be established, the occupied execution and its owners remain
available for a later cleanup attempt. Input consumers that are still active
must not lose their backing allocation during trim or close.

## User outputs in later stages

`GpuTask`, producer callbacks, and publication are still proposed. See the
[canonical design](proposals/user_gpu_pipeline.md) for requested constant uploads,
user-owned scratch/output allocations, and publication-specific dtype/shape/byte
metadata. That design adds bounded host staging and terminal D2H events without
restoring full-image float32 assumptions or framework-managed user device memory.

Stage 1b validation and removal counts are recorded in
[the findings](user_kernel_stage1b_findings_20260926.md).
