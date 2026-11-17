# Prepare real run51 regression inputs

User authorized step1: prepare inputs, including private copying on compute.
Use established results/psana2-gpu-scaling branch and independent Claude review,
maximum three rounds. Base6b87fad3f952a8fdf2e4014846177f362a274cb8.

Implement independent SMD reference generator and verified copying of five full
run51 Jungfrau files and SMDs. Fixed 10k prefix of17872; selected streams3/5/7/8/9.
Audit found83 SlowUpdate gaps in each selected prefix, so required scoped adapter
change supports ordered exact disjoint event ranges. Preserve legacy v1 inputs;
never count gaps as event bytes. No framework/workload/profile changes.

Validate event/step/segment identity, exact ranges and bigdata headers, source/copy
hashes, stable source stat/SMD, private ownership/link count, explicit Lustre
layout, no overwrite and failure-without-complete semantics. Bulk work only on
compute. Record inputs plus generator/helper/decoder provenance. Run offline
integrity tests and independent review before copying; review resulting evidence.
One premium lcls_g allocation, one GPU reserved (unused),8CPUs,32GiB RAM,30min;
prepare only, no performance run, no qualification flag enabled. Source unchanged.
Large artifacts in task scratch; compact records in Git; push reviewed result.

## Recovery after attempt1 timeout

First job61478366 timed out after30:08; no completion receipt, inputs invalid.
Retain failed directory. For attempt2 use new destination and fresh full copying
with45-minute limit; same CPU/memory/account allocation. Sparse header reads use
unbuffered I/O and POSIX_FADV_RANDOM. No reduced correctness or hash checks.
Round2 reviews this change and revised launch scope before submission.

Actual scheduler readback: premium allocated full nodes (128CPU/4GPU) despite
8CPU/1GPU requests. Both jobs use this behavior. Final evidence distinguishes
requested/allocated resources; memory remains32GiB and GPUs unused.
