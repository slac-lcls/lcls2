# Scheduler refresh

2026-11-14 11:20:33 PST: sdfampere036 remains IDLE, CPUAlloc=0,
AllocMem=0 and empty AllocTRES. No jobs returned for this node. Ampere
partition UP, AllowAccounts/AllowQos ALL, no reservation requirement.
This is not proof of immediate job eligibility; preparation's test-only
estimates were Nov30. Refresh again at submission and preserve the exact
node/resources instead of silently changing the experiment.

Existing unrelated jobs are untouched: 45021720 pr299-validation running on
sdfampere012; 45021851 pr299-pressure pending Dependency.
