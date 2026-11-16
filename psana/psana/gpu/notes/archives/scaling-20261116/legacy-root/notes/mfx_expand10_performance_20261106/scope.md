# Single-node performance preparation

Prepare a serial, one-BD/one-GPU runner for the reviewed 10,000-event sample10
fixture. Default: 100 bins, two CUDA streams, nominal batch 7, at least 1,000
warmup events. Support one/two streams and batches 1..16 for later scaling.
Separate clean throughput from opt-in CPU/CUDA/NVTX profiling; retain reviewed
expansion, calibration and accumulation kernels. Check completed event/bin
counts outside the hot loop. Make full export optional and time it separately.

Record the direct-descriptor expansion versus ordinary dense raw gather
distinction. Do not modify psana core, public data or calibration databases.
Do not launch the 10k campaign, MPI scaling, or generate larger data in this
preparation step. Small CPU/device checks and a 1k integration smoke are in scope.

Acceptance: host tests of timing boundaries, generator timing, fixture binding
and failure paths; controlled GPU arithmetic/count rejection; throughput and
profile smoke on the reviewed 1k fixture; restricted Claude review of the exact
implementation commit, response and re-review for blocking findings. Three
review rounds maximum. Branch work/synthetic-mfx-scan, fork monarin/lcls2.
