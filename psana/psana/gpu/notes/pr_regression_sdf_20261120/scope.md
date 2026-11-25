# SDF execution and cluster routing

Implement the shared six-case regression workflow for SDF A100 allocations as
well as Perlmutter. Detect the coordinator cluster, announce the selected path,
default to that cluster, and honor explicit cross-cluster instructions without
silently running locally. Unknown/conflicting identity must stop selection.
SDF NIC receive/transmit observations are mandatory and use physical byte
counters with provenance, reset/duplicate guards and retained raw snapshots.
Keep dataset roots site-specific and validate identical workload identity.
Preserve Perlmutter behavior and historical captures. No production psana changes,
new performance thresholds, automatic qualification, or large data copy/campaign.
Validate offline failure contracts and use small SDF compute probes as available;
distinguish implementation/launch evidence from real-data campaign qualification.
Independent review loop: at most three rounds; existing push authorization applies.
