# Reader screening complete; independent review passed

Job61119210 completed all54 cases on nid001940, lcls_g/gpu_premium, every Slurm
step0:0. Start2026-11-11T00:36:51Z,end02:24:51Z,elapsed6480s. Pacific finish:
November10 at18:24:51PST. Independent receipt audit passes all54cases; exact
source/node acceptance passes. Execution source remains reviewed commit
aca91d45230b3aa18a69f2472e16eebf2f177375.

Claude round3/3 passed native evidence commit
f687d0e2bf1c910e786e9a4e151aab7e61b9330e with no blockers and all required reads
successful. User approved external transfer after the initial automatic approval
block. See review-03.json and responses-03.md. Only documentation changed after
review: finalists, qualified CPU-limit interpretation and closure records.

See RESULTS.md and screening-throughput.png/pdf. Count2 leads at20/40threads;
all5-thread medians are within5% of best, with possible client CPU limits.
Original pooled-ratio rule selects counts1,2,4 for later confirmation. Count8's
5-thread advantage overcount2 is only0.07% with overlapping ranges; no distinct
extra finalist. Screening alone does not establish a final optimum.

Native receipts/raw/accounting/acceptance/hashes are under sweep/. Originals:
/pscratch/sd/m/monarin/psana2-gpu/sim10-striping-20261110/reader-sweep01.
Pilot61118878 evidence is under pilot/, excluded from screening rankings.
Campaign used8807node-seconds, within43200-second budget; this is allocation
time, not premium billing units or current Iris balance.

No confirmation, diagnostic, or psana runs launched. Next proposed phase:
independently placed count1,2,4 copy sets and confirmation per PLAN.md, within
remaining budget. Capture available native readahead limits then (R11).
R8 optional early error cancellation remains deferred. Backend cache state and
remote bytes remain unverified; Lustre counters were unavailable.
