# Verified-copy stage

User authorized creating the first six stripe-count test sets, with an ETA.
Create only count1/2/4/8/16/32 placement1 copies of the five bigdata files,
1MiB stripe size, original pool, automatic OSTs, lcls_g premium on one node.
No read benchmark, confirmation copies or migration of source data.
Implement and review staging before large I/O; review measured completion
receipts afterward. Three-round review limit for this implementation.
Acceptance: all30 destination sizes, independent SHA256 reads and observed
layouts match the accepted source manifest and desired stripe counts; source
identities unchanged; native Slurm job succeeds. Failures leave no completion
marker; partial new copies retained with failed receipt for diagnosis.
