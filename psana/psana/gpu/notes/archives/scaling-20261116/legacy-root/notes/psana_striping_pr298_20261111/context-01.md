Independent plan review round1/3. User wants the parameter list before running
psana2-gpu stripe-read benchmarks, prior512KiB/KvikIO/batch controls, exactPR298.
Scope is planning only. Review PLAN.md and scope.md in this directory, plus
local evidence/pr298.json,pr298-performance.md,pr298-gpu_events.py. Historical
source notes are relative ../sim10_stream_length_20261109/README.md,
../sim10_bulk_target_20261109/README.md,../sim10_100k_bd_sweep_20261109/README.md,
../mfx_kvikio_read_20261108/README.md,and ../sim10_striping_20261110/reader/RESULTS.md.
All required reads are inside this detached worktree. External scratch paths
are provenance/future paths, not required reads. No commands,delegation or edits.

Check that the proposed list is concrete,bounded,and interpretable: distinguish
bulk byte target vs memory budget,CUDA slots vs XTC streams vsKvikIOworkers,
research fix evidence vs exactPRhead67c327a,100kdata vs400k,staging vs cube,
fixed controls vs demonstrated optima. Main screen72cases=6counts×2depths×2BD
counts×3blocks. Ensure SMD metadata/view binding can't silently read source
layout; cache gate and full-global accounting scale to multiBD on oneGPU;
PR build/native correctness gate precedes measurements. Assess candidate
selection,placement confounding,6nodehour cap and later confirmation scope.
No measurements,build or implementation tests have been run in this planning
turn. Source was fetched and relevant APIs inspected. Check plan claims only.
Return structured review at supplied exactSHA with stable finding IDs; denied
required reads or missing crucial evidence are limitations, not automatic pass.
