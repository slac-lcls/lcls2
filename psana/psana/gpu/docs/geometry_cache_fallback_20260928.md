# Rank-local fallback for unseeded geometry

This change follows the Stage 4 publication commit `9730a9982` and addresses an
existing master issue independently of GPU callback scheduling.

Previously, `_pixel_coords()` and `_pixel_coord_indexes()` entered MPI
collectives on a shared-cache miss. A BD-only request for `all_segs=True`, a
different coordinate frame, tilt option, pixel scale, or image offset could
therefore wait for ranks that never requested that variant.

Both accessors now return shared arrays on a hit and compute/cache an unseeded
variant locally on the requesting rank. Cache identity includes detector,
geometry text, segment selection, options, and coordinate/index kind. Repeated
requests reuse that local result. Ranks that never request a variant do not
compute or allocate it. Local copies can increase memory use for variants
requested by many workers; existing shared defaults retain their memory savings.

`RunParallel._setup_jungfrau_shared_caches()` explicitly opts into collective
construction with `_initialize_shared=True`. This internal option requires all
cache ranks to participate. The existing default startup variants remain the
same; automatic GPU geometry and the all-segment warmup are not restored.
SharedCalibcCache and other shared-memory cache protocols are outside this fix.

Validation includes 13 focused tests with real small geometry: numerical values,
dtypes/segment shapes, independent option keys, local reuse, preserved shared
hits, and guards against broadcasts, barriers, or shared allocation on misses.
A three-rank MPI regression exercises the actual shared-startup method and
MPI windows, then requests an unseeded variant on only the leader and only a
follower while other ranks wait. It verifies exact results, preserved shared
defaults, no extra shared windows, and no allocation on the uninvolved rank.
The MPI test has a 90-second timeout and is included in `byhand_*` validation.

Post-commit core, byhand/MPI, and public GPU-task integration results are recorded
under `/sdf/scratch/users/m/monarin/gpu-validation/geometry-cache-fallback-20260928-r1`.
The earlier serial early-close review finding remains a separate open issue.
