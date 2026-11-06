# Jungfrau pedestal workflow chunk-size A/B

User requested reproducing the callback bug on unfixed code and testing whether
PS_SMD_N_EVENTS can work around it. Keep the implementation/review loop active.
Use the documented mfx100848724 run 49 workflow, without claiming it is Mikhail's
latest private command. Run 49 has three physical steps of 1000 events, measured
by direct-file reading of SMD stream 0 (independent of SMD batching/callbacks).

Compare pre-fix 71378658a54dcdd941c9137355a985fa4b637595 with fixed
b60fae55728e7d138747db57fad2c84c10543910 using the same unchanged local native
extensions. Isolated source overlays identify every imported production source.
Exercise real jungfrau_dark_proc stage 1 and MPI stage 2, with one detector panel
via the supported --segind 0 option to bound numerical cost. This tests selection
and pedestal arithmetic, not full-detector performance. Keep outputs under /tmp.

Use 100-event chunks to split steps, 1000 to align one step per chunk, and 10000
to fit the whole run, measuring actual built chunks. Stage 1 uses 10 gate frames
per step. Stage 2 uses nrecs=12, evskip=2, hence expected indices [2:12].
Compare exact timestamps against direct-file oracle before numerical arrays.
Stage 2 gets identical correct gate inputs in every case, so failed stage-1
jobs cannot confound stage-2 counter behavior. Capture natural stage-1 failures.

Instrumentation records callbacks, delivered raw timestamps, chunks and output
arrays; it does not replace the selection predicate or event iteration. Use
explicit data directory and suppress external file-discovery/monitoring only.
Never deploy calibration data. Use read-only detector-name lookup; retain all
deviations and limitations in results. Do not invoke the asynchronous Slurm
wrapper; run stages sequentially so gate-file readiness is deterministic.

No psana core source changes. Validate the actual A/B runs and independently
review the committed harness, evidence and conclusions in a detached checkout.
Default three review rounds and 15-minute reviewer limit. Existing authorization
to send psana code/context to configured Claude and push this branch applies.
