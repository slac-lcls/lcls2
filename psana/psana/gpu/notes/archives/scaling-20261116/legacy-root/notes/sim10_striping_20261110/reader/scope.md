# Reader pilot and initial sweep

User authorized next step: implement standalone CPU reader, validate and review,
then pilot and54-trial screening sweep on the existing verified copies.
Counts1/2/4/8/16/32 x5/20/40threads x3 randomized complete blocks;1MiB requests,
all five files,1MiB stripe size,original pool,one exclusive GPU node per job,
lcls_g premium, no GPU use. Preserve original and copied datasets.
Primary claims limited to verified local-page-cache-cold buffered reads;
backend caches uncontrolled and remote bytes unverified unless usable counters
are established. Separate pilot from ranked sweep. No finalist-copy creation,
confirmation campaign, psana validation or optional diagnostics in this step.
Acceptance: meaningful native reader tests and fault tests, all worker ranges
cover each input exactly once, unchanged file identities/layouts, zero residency
before every trial, fixed physical-core mapping, measured source/binary identity,
all54 receipts plus independent audit and Slurm0:0. No optimum recommendation
until later independent-placement confirmation. Review loop limit3 rounds for
implementation; measured-results review may follow after jobs complete.
