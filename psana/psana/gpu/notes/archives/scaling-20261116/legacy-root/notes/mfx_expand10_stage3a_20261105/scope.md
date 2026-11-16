# Stage 3A scope

Use the existing Stage 2 tiny, tail and 1,000-event fixtures. Add an application-
owned callable with persistent per-CUDA-stream cube banks, consuming the reviewed
sample10 expansion/calibration. Use 30 float32 sum bins and uint64 event counts;
assign bins from the frozen output timestamp -> global event index mapping.
Validate every final cube pixel against float64 CPU sums with a stated rounding
bound, and exact bin counts. Exercise multiple streams, batches, empty bins,
repeated destinations, native closing transitions, run isolation, explicit
completion and abort cleanup. A controlled GPU test checks retention across step
labels; the source has one native step, so native multi-step acceptance is not
claimed. Preserve sums independently of batch.step_generation.

The managed lifecycle/factory/state allocator is still proposed. This bounded
serial pilot creates one task per Run, owns each stream's bank explicitly,
finalizes once after full iteration and joins work before export/cleanup.
Persistent state is capped and separately accounted, outside psana's input budget.
No core API edits, new data generation, DB changes, publication, MPI/GDS or 35 kHz
claims. No scientific threshold, common mode or pixelwise exposure calculation.

Use implement-review-loop, max three rounds, exact commits on
work/synthetic-mfx-scan, base ff5c3304d. Put generated reports/arrays in scratch.
