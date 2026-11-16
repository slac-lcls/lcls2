# Scaling migration scope

Base: 94b741e1388318a69e475481dc93aedb01c2b8b3; branch results/psana2-gpu-scaling.
Source: monarin/lcls2 results/psana-scaling at 9674e3e803354b65c86e42d616af50b23d2c6e3a.
Migrate five campaign records and their source/evidence dependency closure to
GPU docs/scripts/notes. Preserve frozen bytes, original IDs and qualifications;
account for all 359 original added files and inherited dependencies. Integrate
PR study wrappers into the common catalog. Adapt offline arithmetic/catalog
checks and SDF normalized-row replay; validate negative paths, links, archive
hashes, runtime equality and source-only package/test/lint isolation. Use
independent Claude review on an exact commit, at most three rounds.
No runtime changes, new scientific results, new jobs, dataset qualification,
regression template, publishing or source-branch retirement.
