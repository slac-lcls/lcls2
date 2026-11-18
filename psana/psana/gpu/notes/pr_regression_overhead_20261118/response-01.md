# Response to round 1

| ID | Response |
| --- | --- |
| OVR-01 | Fixed. Public collect requires sealed completion/index/summary, no failure marker, final source/input integrity and scheduler audit. Internal unsealed calculation is private. Tests reject failed/unsealed/mutated captures. |
| OVR-02 | Added complete synthetic 24-sample pilot through collect/check_pilot and measured-stage full/minimal checks; reject cache residency/count/history, warmup cache, insufficient CXI, mutation and missing sealing. Fake full native execute was not added: it would replace most MPI/psana machinery and not prove native behavior. The reviewed pilot remains the real observer-path check; no native success claimed yet. |
| OVR-03 | Fixed. Site, reference and manifest require exact pinned artifacts; reference population/hash/bytes also checked against profile. |
| OVR-04 | Fixed. Pilot gate executes after started.json inside failure recording; a rejection writes failed.json. |
| OVR-05 | Implemented. Numbering 1..k<=3, prior failures/final pass, receipt hash, observed residency and final-observation equality are checked. This addresses the new collector's ACCT-09-equivalent limitation. The earlier historical accounting collector is unchanged. |
| OVR-06 | Driver records source directory and declared reviewed SHA. Operator must verify every frozen source against the reviewed Git blob and save the association before submission; this explicit procedural check avoids claiming the supplied SHA alone proves identity. |

All120 offline tests passed. No hardware or overhead equivalence claim.
