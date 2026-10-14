# Kilobyte input candidate: feespec alongside Jungfrau

Discovery on 2026-09-24; no GPU performance measurement in this note.

## Verified data

- Experiment `mfx101210926`, run 387, the existing JF baseline dataset.
- `feespec.raw.hproj`, segment 0: `(2048,) int32`, 8192 bytes per array.
  All first 1000 events contain this array with the same shape and dtype.
- Physical file `mfx101210926-r0387-s000-c000.xtc2`: full event dgrams
  9664–9736 bytes. Ten normal detectors share this stream. The same bytes
  must be included in each comparison, even when only feespec is consumed.
- First 1000 events: 9,735,424 dgram bytes, 10 contiguous event spans.
- First 10000 events: 97,354,744 dgram bytes, 84 contiguous event spans.
  Execution-window and transition fences may further split requests.
- The first 10000 timestamps have SHA256
  `23f9051ac9f98208310171f734251b0ccd1f40fa048d1748553cbd38cce559d6`,
  identical to the existing JF baseline streams 5–9.
- Concatenated first-1000 hproj array bytes have SHA256
  `4447f39348ef54431a1f7084f704f1ae7d868acff7869ae2df9729760b39084b`.

Discovery used SMD offset/size descriptors for 10000 events and CPU-decoded
bigdata dgrams for the first 1000 arrays. Original data were not changed.

## Comparison interpretation

A can keep `gpu_det='jungfrau'` and use normal CPU `feespec.raw.hproj(evt)`
followed by a per-event H2D upload. It needs no new small-detector GPU adapter.
The CPU BigData reader already coalesces adjacent dgrams, so per-event field
extraction does not imply one POSIX read per event. Measure actual counts.

B++ and E can read and parse the complete shared stream on the GPU with a
benchmark-only exclusive-routing override for that specific stream. Record
the original ownership, consume only feespec from that stream, and verify no
duplicate CPU bigdata reads. Preserve all stream bytes and transitions.
This override must not become a silent production routing policy.

For the request-granularity experiment, compare 1000 individual dgrams against
the 10 legal contiguous spans in the first window. One 1000-event transfer
would require reading intervening records too and is a different experiment.
Keep request counts separate from KvikIO worker tasks and actual POSIX calls.

For mixed JF/feespec end-to-end timing, hold JF read scheduling and execution
size fixed when isolating small-stream coalescing. Current E bulk-on also
changes JF reads and admission; a small-stream-only bulk variant must be
explicitly labeled as a benchmark modification. A larger EB input window
must not silently change JF execution sizes or memory pressure.

Use equivalent GPU consumption of hproj values and separately verify CPU/GPU
values, event order and missing-field behavior. Include per-event extraction,
H2D submission, any locator metadata D2H, and final GPU completion in the
appropriate measured path. Avoid concluding E is necessary solely from
fewer I/O calls: compare actual end-to-end time and incremental small-input
cost against a JF-only control in the same allocation.

## Weka FFB preparation

`~/goodstuffs/bashrc` defines `weka_tier` as `weka fs tier location`.
On 2026-09-24, original s000 has 391.85 MB in object storage and only
524.28 KB in SSD read cache. JF source files also have object-store backing
and only partial SSD read-cache coverage. The experiment FFB directories
have no matching run-387/run-51 XTC files.

Use bounded private prefixes in `/sdf/data/lcls/drpsrcf/ffb/users/monarin/`,
staged on a compute node (the login mount is read-only). Verify the actual
staged files with `weka fs tier location` before timing. Staging from the
original source is outside timing. Preserve original offsets/SMD references.

Cold means verified node-page-cache-cold reads from FFB, with file-specific
eviction on the private files and physical NIC counters around the loop.
Warm means verified resident input pages. This does not flush Weka server
caches. Use physical Ethernet counters as in the earlier Weka FFB study;
generic process read_bytes or IB counters alone did not capture Weka traffic.
