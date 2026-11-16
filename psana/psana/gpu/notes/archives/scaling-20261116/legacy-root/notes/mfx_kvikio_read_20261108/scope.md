# Read-only KvikIO fallback ceiling on the accepted 10k sample10 fixture

User asks for a read baseline and settings sweep to find attainable bandwidth
from this dataset to GPU. Keep the existing five JF files on Weka; no staging,
new data, calibration changes, parser/user kernels or psana framework changes.
Branch work/synthetic-mfx-scan, base 4e09db0ec. Use the session's restricted
Claude implementation/review loop, at most three rounds.

Compare exact 50k event-datagram requests with contiguous 4/16/64 MiB file
requests (latter include tiny transition/header overhead). Vary fallback workers,
bounded outstanding requests, task size, pinned bounce-buffer size and optional
opportunistic POSIX direct reads. One GPU/node, 16 allocated CPUs and 64 GiB host
memory; include a four-CPU-affinity control. Each configuration uses a fresh
process. Repeated complete passes target at least 5 seconds, capped at three;
repeat finalists/anchor three times in deterministic shuffled order. Screened
maximum is not a universal hardware limit. Keep total requests and GPU buffer
capacity bounded and actual CPU affinity/settings visible.

Time first submission through all futures completed and final device join,
including Python issue/wait overhead; exclude setup, small warm reads and
post-timing byte witnesses. Validate every future's byte count, total accounting,
bounded buffer ownership, and final GPU-buffer sample bytes against independent
CPU pread. Record manifest identity/file sizes, actual bytes, requested settings,
effective defaults, GPU/node, wall-time definition and cache exposure. No cold
cache, physical Weka-bandwidth, GDS or end-to-end cube-rate claim. No cache flush.
All generated artifacts go to project scratch; compact code/results/review in Git.
