# Raw runner instrumentation overhead

Status: **code reviewed, pilot job61541892 submitted; overhead unqualified**.
Claude round2 passed code `a876e8275c42cfd272d52e5e283f0fe99852dde0` with no blockers.
All120 offline tests passed on that exact commit.

[Scope](scope.md), [driver instructions](../../scripts/performance/regression/OVERHEAD.md),
[review](review-02.json), [response](response-02.md), [validation](validation-03.json),
[frozen source association](production-01.json), [pilot batch script](pilot-01.sbatch).

The 1k/two-pair six-case pilot has24 measured samples plus separate200-event
warmups. It can never qualify overhead. Full qualification requires10k/six pairs
and every loop-time95% interval inside+/-1%. All shipped qualification flags
remain false; production MPI cleanup is deferred.
