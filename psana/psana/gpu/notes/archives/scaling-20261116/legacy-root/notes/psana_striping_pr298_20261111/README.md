# PR298 psana GPU stripe test: reviewed plan ready

[Parameter sweep and controls](PLAN.md) passed independent Claude review at
**e1b818be7fb0af225d0bb47bfb71ae96cb072157**,no blockers.
[Review](review-02.json) and [closure/prerequisites](responses-02.md).

Proposed72fullcases:6stripe counts×1/2CUDAstreams×1/4BDsononeGPU×3blocks.
100k events percase; fixed512KiBbulk,batch100,SMD1000,6GiB/BD,KvikIO8workers,
1MiBtasks,16MiBbounce,compatON,directOFF,1MiBstripes.
PR298execution head:67c327a148a08b2aaf972e67e1e02e367c2b298f.
Four1000eventtimedpilots and separate binding/tail/nativecorrectness gates
precede the screen.6nodehourcap includes allnativepreflightwork.

No new build or benchmark job has run. The user requested the list before
execution; present it and await direction. Next authorized execution work would
first isolate/build the exactPR,adapt/review the harness and data views,then
validate native behavior before submitting the fullscreen. Never reuse the old
research overlay as proof of an exactPR build.

Later optional sensitivity/placement confirmation,multiGPU,multiNode,cube and
400k data generation remain separate stages. Existing100kstripedpayloads are
reused; each needs verifiedSMDmetadata views and actual-open inode checks.
