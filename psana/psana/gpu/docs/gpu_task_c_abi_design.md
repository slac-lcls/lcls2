# GPU Task C ABI — Writing a Task as `int func(void* tc, void* stream)`

**Status:** Proposed — for team review
**Scope:** `psana.gpu` — user-defined GPU work inside the psana2 batch pipeline
**Audience:** users writing compiled GPU analysis code
**Implementation notes:** `gpu_task_c_abi_internals.md`
**Premise:** a task is a **plain C symbol with the signature
`int func(void* tc, void* stream)`**, called **once per L1Accept event and never
for a transition**.  That is the only entry point and the only callback.  Buffer
names, dtypes, and shapes are declared once in the analysis script; the C side
just asks for them by name.

---

## 1. What psana gives you, and what is yours

psana routes DAQ streams to the GPU, reads them with GDS, gathers each event's
segments, and publishes **one** result:

| Key | Contents |
|---|---|
| `'<det>.raw'` | gathered pixels, `PSANA_U16`, shaped `(n_segs, nrows, ncols)` |

plus calibration constants as **device pointers** (pedestals, gain masks), which
psana loads, keeps current across `BeginStep`, and CUDA-IPC-shares between BD
ranks on the same GPU.

Everything after that is your task's: calibration, corrections, geometry, image
assembly, reduction, hit finding.  psana schedules your function inside its own
execution slots, on the slot's CUDA stream, so your kernels overlap its I/O and
CPU deserialisation — and with two subbatches in flight on two slot streams, they
overlap the *other* slot's gather as well.

```
per L1Accept, inside psana's producer (before the event is yielded):

  [1] GDS read + gather        → 'jungfrau.raw'
  [2] YOUR FUNCTION ★          → 'jf_threshold', …
  [3] completion event recorded, outputs leased, host copies started
  [4] event yielded to run.events()
```

---

## 2. The entry point

```c
int userfunc(void* tc, void* stream);
```

* **Called once per L1Accept.**  Transitions — `BeginRun`, `BeginStep`,
  `EndRun` — never reach the task.  There is one call, so there is no phase to
  test and no lifecycle to implement.
* **`tc`** points to a `psana_task_v1` struct (§3): psana's vtable plus an opaque
  `self`.  Valid **only for the duration of the call** — do not store it.
* **`stream`** is the slot's `cudaStream_t`, passed explicitly so the rule "launch
  everything on this stream" is impossible to miss.  Cast it:
  `(cudaStream_t)stream`.
* **Return `PSANA_OK` (0)** on success.  A negative value aborts the run and
  surfaces from `run.events()` as `GpuTaskError`; call `psana_set_error()` first
  to attach a message.  Positive values are reserved.

### 2.1 Declaring your buffers

Names, dtypes, and shapes live in the analysis script, so psana knows the exact
per-event VRAM cost before it submits anything and can size subbatches correctly:

```python
from psana import DataSource, GpuTask

peaks = GpuTask('libuser.so:userfunc',
                inputs  = ['jungfrau.raw'],
                outputs = {'jf_threshold': ('f4', 'same'),        # published
                           'n_hits':       ('f4', ('n_segs',))},
                scratch = {'jf_calib':     ('f4', 'same')})       # not published

ds  = DataSource(exp='mfx100852324', run=77,
                 gpu_det='jungfrau',          # route + gather + load constants
                 gpu_fn=peaks)
run = next(ds.runs())
```

| Field | Meaning |
|---|---|
| `inputs` | keys the task reads.  Validated before the run starts, so a typo fails at setup with the available-key list rather than on event 0 |
| `outputs` | `name → (dtype, shape)`.  Allocated per event, published as results, and prefetched to the host (§8) |
| `scratch` | same allocation and sizing, **not** published.  For intermediates |

In C both are reached with the *same* call — `psana_buffer(name, &view)` — and
which dict a name appears in is what decides whether psana publishes it, copies it
to the host, and shows it to the event loop.

A **shape spec** is either:

* `'same'` — the shape of the first entry in `inputs`, *for this event*, so a
  ragged event (fewer segments because a stream had no data) is handled with no
  effort on your side;
* a tuple of ints and dimension names — `('n_segs',)`, `('n_segs', 'nrows')`, or
  literals like `(2, 1024)`.  Names refer to the first input's dimensions.

Because psana knows every shape and dtype, it computes the per-event byte cost
itself: there is no `bytes_per_event` for you to work out, keep in sync with your
kernels, or get wrong when a run has fewer DAQ streams.

`'<det>.raw'` is the only name psana reserves; everything else is yours,
including a name like `'jungfrau.calib'`, which carries no special meaning.  Every
buffer the task uses must be declared — `buffer()` on an undeclared name is an
error, not a lazy declaration.

```python
for ctx in run.events():
    thr = ctx.get('jf_threshold').on_cpu        # already host-resident (§8)
    raw = ctx.get('jungfrau.raw').on_cpu        # psana's, if you want it too
```

A list of tasks is a pipeline, run in order, and a later task may read an earlier
one's output by name:

```python
gpu_fn=[calibrate, find_peaks]                 # two GpuTask objects
```

### 2.2 Setup and per-run state

psana holds no state for you.  Anything that must persist between events lives in
your own file-scope variables — the task is called from a single producer thread,
never concurrently, so no locking is needed:

```c
static int      g_ready = 0;
static uint32_t g_run   = 0;
static MyParams g_params;

/* … inside the event call … */
if (!g_ready || g_run != psana_run_number(tc)) {
    g_run = psana_run_number(tc);
    g_params.threshold = 5.0f;
    g_ready = 1;
}
```

Four things to know:

* **First-call work may block**, and psana accepts that: NVRTC compilation or a
  cuFFT plan synchronises, so the first event of the run pays a one-time stall
  inside the producer.  §7 rule 2 forbids blocking on *every* event, not on the
  first.
* **There is no teardown call.**  Nothing runs at end of run, so free nothing and
  reuse everything; detect a new run with `psana_run_number()` as above if your
  state is run-dependent.
* **File-scope state means one symbol per task.**  Two `GpuTask` entries pointing
  at the same symbol would share those variables.  Export two symbols instead.
* **There is no psana-allocated persistent device memory in v1.**  Every buffer
  psana hands you is per event.  A task needing its own long-lived table — a mask,
  a lookup table — allocates it once in first-call setup with `cudaMalloc`, and
  that allocation is outside psana's VRAM budget, so keep it modest and account
  for it yourself.  If this becomes common the answer will be a `persistent=`
  entry in the `GpuTask` spec, not a runtime call.

### 2.3 There is exactly one position in the per-event chain

Tasks run once `'<det>.raw'` is complete for the event, and before the event is
yielded.  If a detector's segments are split between CPU and GPU, psana merges the
raw partials first, so you always see the complete `'<det>.raw'`.

---

## 3. The ABI

`psana_gpu_task.h` is **pure C99**: it includes only `<stdint.h>` and
`<stddef.h>`, needs no CUDA headers, and declares no symbols to link against —
every call goes through the vtable psana hands you.  Your `.so` therefore has no
undefined psana symbols and no build coupling beyond the header.

```c
#define PSANA_TASK_ABI_V1 1

typedef enum {
    PSANA_F32 = 0, PSANA_F64 = 1, PSANA_U16 = 2, PSANA_I32 = 3, PSANA_U8 = 4
} psana_dtype;

enum {
    PSANA_OK         =  0,
    PSANA_ERR_NO_KEY = -1,   /* no such input for this event */
    PSANA_ERR_NAME   = -2,   /* name was not declared in the GpuTask spec */
    PSANA_ERR_BUDGET = -3,   /* VRAM budget exhausted */
    PSANA_ERR_ABI    = -4
};

typedef struct {
    void*    ptr;            /* device pointer */
    int64_t  shape[4];
    int32_t  ndim;
    int32_t  dtype;          /* psana_dtype */
    uint64_t size;           /* element count */
} psana_view;

typedef struct psana_task_v1 {
    uint32_t abi_version;    /* == PSANA_TASK_ABI_V1 */
    uint32_t struct_bytes;   /* sizeof(*this) as psana built it — §10 */
    void*    self;           /* opaque psana state */

    /* ── this event's input, by declared key ─────────────────────────────────── */
    int      (*input)(void* self, const char* key, psana_view* out);

    /* ── this event's output or scratch buffer, by declared name ─────────────── */
    int      (*buffer)(void* self, const char* name, psana_view* out);

    /* ── constants psana owns, refreshed at BeginStep — fetch every event ───── */
    int      (*calib_const_device)(void* self, const char* det, const char* name,
                                   psana_view* out);

    /* ── identity ───────────────────────────────────────────────────────────── */
    uint64_t (*timestamp)(void* self);
    uint32_t (*run_number)(void* self);

    /* ── errors ─────────────────────────────────────────────────────────────── */
    void     (*set_error)(void* self, const char* msg);
} psana_task_v1;
```

Six function pointers and three data fields.  No phase enum, no `declare_*`, no
`stream()` accessor, no state slot, and no key introspection — the stream is a
parameter, the shapes and names are in the script, and state is yours.

The header supplies one `static inline` wrapper per member, so task code never
writes `->self`:

```c
static inline int psana_input(void* tc, const char* key, psana_view* v)
{ const psana_task_v1* t = (const psana_task_v1*)tc; return t->input(t->self, key, v); }

static inline int psana_buffer(void* tc, const char* name, psana_view* v)
{ const psana_task_v1* t = (const psana_task_v1*)tc; return t->buffer(t->self, name, v); }

static inline int psana_abi_ok(void* tc)
{ return tc && ((const psana_task_v1*)tc)->abi_version == PSANA_TASK_ABI_V1; }
/* … one per member … */
```

What comes back:

* `input()` views are **C-contiguous device memory**, `ndim ≤ 4`.  psana's own
  `'<det>.raw'` is `PSANA_U16`; every other key is an earlier task's output, with
  the dtype that task declared.  `PSANA_ERR_NO_KEY` means the detector had no data
  for this event — that is normal, not a failure.
* `buffer()` returns **this event's own non-overlapping memory** for a declared
  name, with the shape and dtype from the spec, resolved against *this* event's
  input shape when the spec is `'same'`.  Contents are **undefined** on entry
  (recycled memory): write every element you publish.  A name declared in
  `outputs` is published and copied to the host; one declared in `scratch` is
  neither, and may be reused by other events in the subbatch.
* `calib_const_device()` returns psana's device-resident constants for the current
  step.  With several BD ranks on one GPU these are already IPC-shared from a
  leader rank — use them, do not copy them per rank.

---

## 4. The C++ wrapper

Over a C vtable, method syntax costs a header and nothing else:
`psana_gpu_task.hpp` is **C++11, no pybind11, no libpython, no CUDA
requirement**.  It is the recommended way to write a task in C++ or CUDA C++.

```cpp
namespace psana {

template <class T> struct View {
    T*      ptr  = nullptr;
    int64_t shape[4] = {0,0,0,0};
    int     ndim = 0;
    size_t  size = 0;
    explicit operator bool() const { return ptr != nullptr; }
};

class Event {
public:
    explicit Event(void* tc) : t_((psana_task_v1*)tc) {
        if (!psana_abi_ok(tc)) throw std::runtime_error("psana task ABI mismatch");
    }
    template <class T> View<T> input (const char* key);   // throws if absent
    template <class T> View<T> get   (const char* key);   // empty if absent
    template <class T> View<T> buffer(const char* name);  // shape from declaration
    template <class T> View<T> calib_device(const char* det, const char* name);

    uint64_t timestamp()  const;
    uint32_t run_number() const;
private:
    psana_task_v1* t_;
};

} // namespace psana
```

`View<T>` checks the declared dtype against `T` and throws naming the buffer on
mismatch, so a `uint16_t` view of a float32 result is an error rather than silent
garbage.  **Exceptions must not cross the ABI**, so wrap the body:

```cpp
extern "C" int userfunc(void* tc, void* stream) {
    try { return run(tc, (cudaStream_t)stream); }
    catch (const std::exception& e) { psana_set_error(tc, e.what()); return PSANA_ERR_ABI; }
}
```

---

## 5. A complete task

Jungfrau calibration from `'<det>.raw'` plus thresholding.  One function, no
lifecycle, no state object, no shape arithmetic.

```cpp
#include <psana_gpu_task.hpp>

static int      g_ready = 0;
static uint32_t g_run   = 0;
static MyParams g_params;

static int run_event(void* tc, cudaStream_t stream)
{
    psana::Event evt(tc);

    auto raw = evt.input<uint16_t>("jungfrau.raw");
    if (!raw) return PSANA_OK;                        // no data this event → skip

    if (!g_ready || g_run != evt.run_number()) {      // first event, or a new run
        g_run = evt.run_number();
        g_params.threshold = 5.0f;                    // compile kernels / plans here
        g_ready = 1;
    }

    // psana refreshes these at BeginStep, so fetch them per event, do not cache
    auto peds  = evt.calib_device<float>("jungfrau", "pedestals");
    auto gmask = evt.calib_device<float>("jungfrau", "gain_mask");

    auto calib = evt.buffer<float>("jf_calib");       // declared in scratch
    auto out   = evt.buffer<float>("jf_threshold");   // declared in outputs

    jungfrau_calib<<<l, m, n, stream>>>(raw.ptr, peds.ptr, gmask.ptr,
                                        calib.ptr, raw.size);
    threshold     <<<l, m, n, stream>>>(calib.ptr, out.ptr, g_params, raw.size);
    return PSANA_OK;      // psana records the completion event, leases the output,
                          // and starts the pinned host copy — §8
}

extern "C" int userfunc(void* tc, void* stream) {
    try { return run_event(tc, (cudaStream_t)stream); }
    catch (const std::exception& e) { psana_set_error(tc, e.what()); return PSANA_ERR_ABI; }
}
```

```python
import cupy as cp
from psana import DataSource, GpuTask

ds  = DataSource(exp='mfx100852324', run=77, gpu_det='jungfrau',
                 gpu_fn=GpuTask('libuser.so:userfunc',
                                inputs  = ['jungfrau.raw'],
                                outputs = {'jf_threshold': ('f4', 'same')},
                                scratch = {'jf_calib':     ('f4', 'same')}))
run    = next(ds.runs())
stream = cp.cuda.Stream(non_blocking=True)      # your stream, for GPU follow-up work

for ctx in run.events():
    res = ctx.get('jf_threshold')               # a GPUResult

    # (a) host copy — free if the pipeline already prefetched it (§8),
    #     otherwise one blocking D→H, cached either way
    thr = res.on_cpu                            # np.ndarray

    # (b) independent D→D copy on the device — always safe, and the slot may
    #     recycle immediately afterwards.  Costs ~2 ms for a Jungfrau frame
    thr_gpu = res.on_gpu                        # cp.ndarray

    # (c) zero-copy view into the slot buffer — the fastest GPU path.  Every
    #     kernel reading it must run on `stream` and be enqueued inside the
    #     block; __exit__ records the done-event psana needs to recycle the slot
    with res.on_gpu_view(stream) as thr_view:   # cp.ndarray, no copy
        my_followup_kernel(thr_view, stream=stream)
```

Pick one per result and know the trade: `on_cpu` if the answer leaves the GPU,
`on_gpu_view` if it stays and you care about the copy, `on_gpu` if you want device
data without thinking about lifetimes.  **`on_gpu_view` requires
`gpu_d2h_chunk_size = 0`** — it raises once psana has scheduled an automatic host
copy for that result, because the slot is then already promised to the pipeline.
`on_gpu` and `on_cpu` work either way.

To publish the calibrated array as well, move `jf_calib` from `scratch` to
`outputs` in the script — **the C code does not change at all.**  Geometry is
yours too: psana assembles no images, so a detector-frame picture means your own
kernel publishing `jf_image`, or host-side work in the event loop.

---

## 6. Memory

* `buffer()` hands out a slice of a **per-`(slot, name)` arena** whose bump offset
  resets each subbatch.  Every event gets distinct memory, because all events in a
  subbatch are still live when the loop reads them.  The mechanism is in
  `gpu_task_api_internals.md` §2.
* **Zero `cudaMalloc` in steady state** — psana sizes each arena from your
  declarations and it stops growing.  This is why allocating your own per event is
  forbidden: it reproduces a CuPy-pool fragmentation OOM this codebase already
  fixed once.
* A `'same'` spec resizes per event, so ragged output is automatic.
* Constants come from `calib_const_device()` (psana's, IPC-shared across ranks).
  A long-lived table of your own is not psana's to give in v1 — see §2.2.

---

## 7. Rules

1. **Launch on the `stream` argument.**  psana's slot streams are non-blocking, so
   they do *not* serialise against the legacy default stream.  A launcher
   hardcoded to the default stream has to be bracketed with events by its caller,
   which funnels every slot through one stream and defeats double buffering — a
   stopgap, not a design.
2. **No host synchronisation, except once on the first event.**  No
   `cudaDeviceSynchronize`, no `cudaStreamSynchronize`, no blocking `cudaMemcpy`,
   no `printf` of device values.  The task runs inside the producer, so a stall
   there stalls the gather, the pre-issued KvikIO read, and the CPU event loop at
   once.  First-call setup (§2.2) is the documented exception.
3. **Do not copy results to the host yourself** — declare the output and let §8
   do it.
4. **Do not record or wait on your own completion event.**  psana records one
   after the last task and hangs a lease on every output.
5. **No `cudaMalloc` per event.**  Every per-event buffer comes from `buffer()`.
   A one-time allocation in first-call setup is fine, and is outside psana's
   budget (§2.2).
6. **Fetch `calib_const_device()` every event.**  There is no begin-step callback,
   so a cached pointer goes stale silently when constants are refreshed.
7. **Link `libcudart` dynamically**, against the same CUDA major version CuPy
   uses.  A statically linked cudart is the usual cause of a valid slot stream
   handle being rejected.
8. **Single-threaded, non-reentrant.**  psana calls the symbol from one producer
   thread, never concurrently, and releases the GIL across the call — so do not
   assume you hold it.  If your library embeds Python, acquire it yourself.
9. **`tc` is call-scoped.**  Nothing reachable from it outlives the call — not the
   handle, not a view, not a pointer inside one.

---

## 8. Getting outputs to the host

Every declared output gets psana's pinned copy pipeline: a ring of pinned host
staging slots, transfers issued in chunks on a dedicated stream as soon as the
batch's completion event fires, overlapped with the *next* subbatch's kernels and
with the CPU event loop, and — when every key for an event is host-backed — the
slot released *before* the event is yielded.  By the time the loop reaches
`ctx.get('jf_threshold').on_cpu`, the data is there and the call is a cache hit.

Nothing a task can do from inside the call matches that: the task runs while the
producer is on the critical path, and a transfer it issues cannot outlive the call
that issued it.

Configuration is psana's, not the task's — `gpu_d2h_chunk_size > 0` enables the
pipeline.  Buffers declared as `scratch` are never copied, which is the reason to
declare an intermediate that way.

**If your consumer is on the GPU, the prefetch works against you.**  Once the
pipeline has scheduled a host copy for a result, `on_gpu_view()` raises: psana may
release the device slot as soon as that copy lands, so a zero-copy view into it
cannot be handed out.  Two ways out, both in the loop rather than in the task:

* run with `gpu_d2h_chunk_size = 0` — no prefetch, and `on_gpu_view()` gives a
  zero-copy view of every result;
* or keep the prefetch and use `on_gpu`, which returns an independent D→D copy
  that outlives the slot.

Declaring a buffer as `scratch` instead of an output is the third option, when
nothing outside the task needs it at all.

---

## 9. Errors and skipping

* `PSANA_OK` with no `buffer()` call = this event has no output for this task.
  `ctx.get('jf_threshold')` then raises `KeyError` for that event.
* A negative return aborts the run: psana synchronises the slot stream (so no
  in-flight kernel is still writing into a buffer that may be freed), wraps the
  code and any `set_error()` message as
  `GpuTaskError(task=…, code=…, timestamp=…)`, and leaves the slot empty rather
  than half-occupied.  Fail-fast is deliberate — a partially written result
  published under your name is indistinguishable from a correct one.
* Accessor failures are diagnosable without a debugger: each returns a distinct
  code, and psana names the key or buffer in the exception.  `PSANA_ERR_NAME`
  means the C code asked `buffer()` for a name the script did not declare, and
  psana lists the declared names.

---

## 10. Building

```
psana/psana/gpu/include/psana_gpu_task.h     ← C99 ABI
psana/psana/gpu/include/psana_gpu_task.hpp   ← C++11 wrapper (optional)
```

Both install inside the Python package, so the headers always match the psana you
will import.  Find them with `psana.gpu.get_include()`:

```bash
PSANA_INC=$(python -c 'import psana.gpu; print(psana.gpu.get_include())')

nvcc -std=c++11 -Xcompiler -fPIC -c kernels.cu -o kernels.o
nvcc -std=c++11 -Xcompiler -fPIC -I$PSANA_INC -c userfunc.cu -o userfunc.o
nvcc -shared userfunc.o kernels.o -o libuser.so -lcudart
```

Nothing from psana is linked, and the `.h` needs no CUDA headers — only your own
translation units do.

**Version compatibility.**  `psana_abi_ok(tc)` (and the `psana::Event`
constructor) compares the `PSANA_TASK_ABI_V1` baked into your binary against the
running psana's `abi_version`, and fails with a message naming both numbers plus
"rebuild your extension against the installed psana headers".  Additive changes to
the vtable do *not* require a rebuild: psana reports its own `sizeof` in
`struct_bytes`, and an older task never reads past what it knows.  To use a newly
added optional member, guard it:

```c
if (t->struct_bytes >= offsetof(psana_task_v1, new_member) + sizeof(t->new_member))
    t->new_member(t->self, …);
```
