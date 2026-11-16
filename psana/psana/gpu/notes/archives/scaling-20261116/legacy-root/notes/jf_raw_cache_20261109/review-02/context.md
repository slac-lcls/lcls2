Second independent review: preparation/failure-classification fixes and available
runtime evidence. This is NOT final performance acceptance; final job43987249
is running on sdfampere002. Review exact supplied SHA, base319d1c8ce. Prior
readiness review585111a45 passed with JFRC-01..07 non-blocking findings; read
review-01/review.json and response.md. No nested review/delegation.

Review new code since585111a45, especially common.condition, reused warm_cache.py
and memory_state.py, common.cache_unavailable, bench/read handling of warm-only
CacheGateError (MPI exit42), campaign partial outcomes, and analysis/audit.py.
Cache threshold remains zero cold/>=99.9% EACH FILE warm. The final retry allows
at most8 missing-page repair passes, not unbounded retries or weaker thresholds.
Warm reset/interleaved preparation is untimed; all snapshots persist on failure.
Other failures must abort; skipped warm gates must never receive rates or pass
as completed warm measurements. Verify signal-aware cleanup and actual launcher
regression tests; SIGKILL remains uncatchable. Final allocation90min, same20
cases/event counts and unchanged raw workload. Fifteen CPU tests pass; tests.log
and test_launcher.py are supplied. No production psana or kernels modified.

Runtime evidence INSIDE this worktree at build/jf-review-evidence/r1 and r2,
with matching r1-logs/r2-logs and copy-index.json (coordinator byte-verified).
R1 job43983799/sdfampere038 ran code1e526c602: all source/copy bigdata SHA and SMD
references passed, 1BD/4BD200-event raw-pixel preflights passed, one full Weka1BD
cold case passed at5.616792GiB/s,335571760000 bytes,zero initial residency,
334.717956GiB physicalNICRX. Then warm gate failed BEFORE timing. That first
failure lacked final residency vectors; this evidence gap is fixed. Local xtc
cleanup recorded true; root/compiler-cache cleanup could not be checked after
allocation (SSH access denied without active job and host-key mismatch; no
security setting changed). Do not mix this single cold rate with another node.

R2 job43985860/sdfampere032 ran code13e576f33: new initial preparation probe failed
before local copying after3 repairs. Cache JSON records five file percentages:
prepared99.9652,99.9951,99.9990,99.9943,81.7443; last file improved95.5539,98.2636,
99.0872. Other files stayed stable. This monotonic progress motivates bounded8
repairs in R3. R2 cleanup.json and root-cleanup.json both record removal. No R2
GPU rates exist. No claim the node change proves NUMA policy caused improvement.
Raw copied artifacts have source hashes from their execution commits, not this
new code; coordinator preserves them unchanged. You cannot execute/hash/query
live resources; report that limitation. Current source APIs remain inspectable.

Required outcome is code/evidence review of fixes and previous findings, not
approval of a final reproduction. R3 results need a final independent review
when available. Read README/scope with this latest context: portions of README
still describe historical attempts and final summary is intentionally pending.
