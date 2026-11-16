Review the four-worker Weka NIC/cache benchmark addition, independently and read-only.
Base: 07df579a3. Implementation: 5e878baa84893537fd614d235d0ae57192197133
(the actual reviewed SHA is supplied by the helper; subsequent commit may add evidence).

Scope: notes/mfx_kvikio_network_20261108/scope.md. Read the new README, campaign,
summarize, test_network, run.sbatch; modified read_case.py and new network_watch.py
under notes/mfx_kvikio_read_20261108. Reference unchanged plan_reads.py/audit.py
for buffer ownership/fixture validation. Do not review unrelated repository code.
This is an explicitly authorized cold/warm read test of five owner-controlled
synthetic fixture files, not production source data; file-specific cache eviction
is authorized. No global cache flush or data mutation. Four workers/four logical
CPUs, one A100, exclusive Slurm allocation. Runtime settings frozen by existing
runtime wrapper. Historical method used ethtool physical counters and zero mincore
before reads; ordinary counters had missed Weka traffic.

Acceptance: exact 50k requests/33,582,000,000 bytes per pass; drain futures before
buffer reuse/closing, forced fallback/effective config, sampled GPU-vs-CPU witnesses,
zero cache residency before cold passes; physical NIC endpoints/deltas/rates per
port; no unsupported exact cache hit ratio or physical device/backend ceiling claim.
Read timing and NIC timing have distinct envelopes. Warm means repeated reads;
cache residency is measured. All NICs remain separate; node-wide traffic, readahead
and protocol bytes limit precise attribution. Check method correctness and whether
reported conclusions follow evidence. Sampling overhead can reduce NIC-envelope
bandwidth relative to application-envelope bandwidth; this must be disclosed.

Coordinator test logs and results.json are supplied. Native reports and scheduler
logs are in build/network-evidence in this detached worktree, with copy-index.json
recording coordinator-verified byte-identical copies and hashes. Reviewer cannot
execute tests/hash files/query live GPU with allowed tools; state this limitation.
No nested reviewer/agent loop, shell, edits, deployments or human messaging.
Return structured findings, stable IDs, severity and blocking status. Up to three
review rounds. Do not demand GPU reruns just for doc/metric-name improvements.
