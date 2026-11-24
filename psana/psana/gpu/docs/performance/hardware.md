# Hardware and cache semantics

Hardware is part of the benchmark configuration. The same GPU family does not
imply the same SSDs, PCIe links, NUMA placement or I/O limits. Observations below
are dated snapshots, not scheduler availability or a guarantee that hardware
and tuning remain unchanged.

## Ampere node inventory — October 26, 2026

**20 of 42 nodes collected.** This was a read-only inventory, not a bandwidth
measurement. [CSV](evidence/ampere_inventory_20261026.csv) and
[JSON](evidence/ampere_inventory_20261026.json) preserve per-node evidence,
collection timestamps/jobs, firmware, capable and negotiated links, NUMA and
settings. Uncollected nodes remain unknown in this inventory.

| Profile | SSD model (two drives/node) | TB/drive | Firmware | Active link/SSD | Nodes (sdfampere suffix) |
| --- | --- | ---: | --- | --- | --- |
| S1 | `WUS4BB076D7P3E3` | 7.68 | R1110021 | Gen3 x4 | 002, 004, 026, 028 |
| S2 | `KCM6XRUL7T68` | 7.68 | 0106 | Gen4 x4 | 010, 011, 013, 014, 015 |
| S3 | `KCM6XRUL7T68` | 7.68 | 0106 | Gen4 x4 | 012 |
| S4 | `KCM6XRUL7T68` | 7.68 | 0105 | Gen4 x4 | 016, 017, 018, 019, 020, 023 |
| S5 | `KCD6XLUL3T84` | 3.84 | 0102 | Gen4 x4 | 031, 033, 034 |
| S6 | `SAMSUNG MZQL27T6HBLA-00A07` | 7.68 | GDC5602Q | Gen4 x4 | 038 |

Capacities use decimal TB. S1 is the WD model; S2/S3/S4 are Kioxia CM6;
S5 is Kioxia CD6; S6 is Samsung. All collected scratch arrays were RAID0 with
512-KiB chunks and SSD NUMA node 5. S2 and S3 deliberately remain separate:
012's observed request limits differed despite matching SSD model/firmware.

| Profiles | md maximum request | SSD maximum request | md read-ahead |
| --- | ---: | ---: | ---: |
| S1, S2, S4, S5, S6 | 512 KiB | 1,280 KiB | 4,096 KiB |
| S3 (012) | 128 KiB | 128 KiB | 4,096 KiB |

Request limits, read-ahead, scheduler settings, negotiated link state and CPU
binding can change; record them again in each allocation. Do not turn profile
labels into permanent hardware guarantees. In particular, 035 was uncollected
in this snapshot, so its historical scaling rate does not establish a matching
SSD profile. The earlier [012/027 inspection](../../notes/lscratch_numa_012_027_20261022.md)
linked a large cold-rate difference to Gen4 Kioxia versus Gen3 WD storage;
that is separate evidence from this inventory.

## Hardware used by the recent campaigns

| Observation | SDF Ada 016, job 42114353 | Perlmutter nid008517, job 60415698 |
| --- | --- | --- |
| Observation date | October 27, 2026 | October 28, 2026 |
| GPU | Ten L40S installed; GPU0 used, 46,068 MiB reported | Four A100-SXM4-80GB installed; GPU0 used |
| Host RAM | About 755 GiB; 700 GiB requested | About 251 GiB; allocation details in inventory |
| Input stage | XFS `/lscratch` on six-drive RAID0 (`md0`), about 23.04 TB filesystem | Private files on Lustre `/pscratch`; no usable local NVMe/SSD stage |
| Scratch SSDs | Six Samsung `MZQL23T8HCLS-00A07`, about 3.84 TB each, firmware `GDC5A02Q` | Backend devices not exposed as local block devices |
| Storage connectivity | SSD endpoints Gen4 x4; Gen5 x16 switch uplinks shared with GPUs | Slingshot NICs to Lustre/OSS; host-memory staging before H2D |
| GPU PCIe | Gen4 x16 capable; idle inventory sampled Gen1 x16 | GPU0 Gen4 x16; see actual node inventory |
| Placement | SSDs span NUMA domains 0/1/3; GPU0 on domain 3 | GPU0/NIC proximity and CPU binding recorded; use actual-node evidence |

[Compact campaign hardware evidence](evidence/hardware_campaigns_202610.json)
contains actual allocation observations and source hashes. The L40S idle link
sample must not be presented as a measured transfer-time link rate.

**Ada 008 is a different observation:** its [raw-read qualification](../../notes/ada008_raw_read_20261027.md)
used six **Micron 7450** drives (`MTFDKCC3T8TFR`), Gen4 x4 endpoints and Gen5 x16
uplinks. Do not attribute that SSD model or its raw-read bandwidth to Ada 016.
The [Ampere 013 qualification](../../notes/ampere013_qualification_20261026.md)
provides measured rates for one inventory profile; the inventory itself does not.

## Perlmutter network and Lustre capacity — November 23, 2026

A GPU node has **100 GB/s of aggregate raw NIC capacity**. The saved Lustre
configuration on the inspected node selects one NIC, implying a **25 GB/s raw
ceiling for that configured filesystem path**. Neither number is a sustained
application read rate. Decimal GB/s uses `10**9` bytes/s; 25 GB/s is 23.283 GiB/s.

### Documented capacity and observed configuration

[NERSC's architecture documentation](https://docs.nersc.gov/systems/perlmutter/architecture/)
(accessed November 23, 2026) specifies four 200 Gb/s Slingshot NICs per GPU node,
one per CPU-only node, and aggregate scratch bandwidth of 7.8 TB/s reads and
6.6 TB/s writes. The scratch figures describe the shared filesystem as a whole;
they are not a per-node entitlement. The separate
[scratch guide](https://docs.nersc.gov/filesystems/perlmutter-scratch/) summarizes
aggregate bandwidth as greater than 6 TB/s.

Read-only SSH diagnostics from login10 reached **nid200276**, in the user's
existing job **61889954**. At 23:25:36 UTC, the node reported four A100-SXM4-80GB
GPUs; `hsn0`–`hsn3` were up at 200,000 Mb/s. `cxi_stat` reported all four Cassini
links up at `BS_200G`, each with a negotiated 16.0 GT/s PCIe x16 link.
`/pscratch` was mounted as Lustre with `@kfi` endpoints. The subsequently read
`/etc/lnet.conf` selected:

```yaml
net:
    - net type: tcp42
      local NI(s):
        - interfaces:
              0: hsn0
    - net type: kfi
      local NI(s):
        - interfaces:
              0: cxi0
```

Both entries refer to the same physical NIC. For this saved configuration,
concurrent Lustre readers share its 25 GB/s line rate; adding BDs or GPUs does
not automatically spread filesystem traffic over four NICs. Runtime inspection
with `/usr/sbin/lnetctl net show` and `/usr/sbin/lctl list_nids` was denied access
to `/dev/lnet`. Thus the single-NIC conclusion is **configuration-based**, not a
verified runtime rail inventory or a traffic measurement. This observation does
not establish the configuration of the historical campaign nodes.

All four NICs connect to the Slingshot fabric and can support inter-node
application communication. MPI and GPU communication libraries use that fabric;
see NERSC's [MPI networking guidance](https://docs.nersc.gov/performance/network/),
[NCCL integration](https://docs.nersc.gov/development/containers/shifter/how-to-use/),
and [NVSHMEM guide](https://docs.nersc.gov/development/programming-models/mpi/nvshmem/).
Actual NIC selection depends on the library and placement/configuration. This
inspection does not establish an exclusive split of one storage NIC and three
MPI NICs: `cxi0` may also carry application communication. NERSC's reason for
selecting one Lustre NIC, and support for using additional NICs, remain unconfirmed.

[Compact diagnostic evidence](evidence/perlmutter_network_20261123.json)
retains commands, selected output, normalized link observations and limitations.
It was transcribed from this session's tool output; it is not a complete raw
terminal transcript. No new throughput benchmark, allocation, or configuration
change was performed, and no independent review is claimed.

### Context for the measured read rates

| Campaign/profile | Node | Reported GiB/s | Decimal GB/s | Fraction of a 25 GB/s link |
| --- | --- | ---: | ---: | ---: |
| [Application S stage](campaigns/perlmutter/sim10-xpp-cube/2026-11-16-se-ec-ecc-100k/README.md), 4 BDs / 4 GPUs / 16 workers | nid001020 | 5.181 | 5.56 | 22.3% |
| [Staging scaling](campaigns/perlmutter/sim10-staging/2026-11-16-gpu-scaling/README.md), count 1 / 16 BDs / 4 GPUs / 64 workers | nid001008 | 9.842 | 10.57 | 42.3% |
| [C++ reader](campaigns/perlmutter/cpp-read/2026-11-11-stripe-screen/README.md), count 2 / 20 threads | nid001940 | 8.317 | 8.93 | 35.7% |
| C++ reader, count 2 / 40 threads | nid001940 | 12.916 | 13.87 | 55.5% |

Fractions are illustrative arithmetic against a single-link line rate, **not
measured NIC utilization**. These campaigns differ in node, concurrency, access
pattern, CPU placement and work performed; their ratios do not isolate software
overhead. C++ reads full files into host buffers. The application S stage
includes seed staging/delivery to GPU, drain, an MPI barrier and callback memory
queries, using KvikIO CPU fallback. Its rate denominator is requested seed bytes,
not expanded images or NIC bytes. Its NIC fields are unavailable; the separate
staging campaign retained aggregate Cassini receive counters.

The application campaign's four-BD/one-GPU bridge reached 5.208 GiB/s, close to
its four-BD/four-GPU result. The separate 16-BD run shows that approximately
5 GiB/s is not a universal psana staging ceiling. More GPUs alone did not improve
the four-BD control. Worker concurrency, request handling, host-to-GPU copies,
scheduling, storage and contention remain possible limits; these comparisons
have not isolated the dominant cause. NERSC's
[Lustre striping guide](https://docs.nersc.gov/performance/io/lustre/) explains
another factor in usable I/O bandwidth. No guaranteed sustained per-node Lustre
rate was found in the documentation consulted.

## What cold and warm mean

| Condition | Prepared state | Timed input path and interpretation |
| --- | --- | --- |
| Ada cold | Evict job-owned measured file ranges; require at most 1% host-page residency | Local NVMe RAID → host memory → GPU; does not establish cold SSD-controller caches |
| Ada warm | Prefill measured file ranges; require at least 99% residency before/after | Host file cache → GPU, still including reads, copies, parsing/gather and the selected task; not GPU-resident input |
| Perlmutter cold | At most 1% client residency on private Lustre prefixes before each sample | Lustre/OSS → network → host memory → GPU; server/OSS/storage caches and shared-storage traffic remain uncontrolled |

Perlmutter had no `/lscratch`-backed local SSD dataset for this campaign.
`/tmp` and `/dev/shm` were memory-backed; the job's `TMPDIR` was on Lustre.
No Perlmutter warm result is claimed. Cache state and kernel/JIT warmup are
separate controls. All paths above used KvikIO CPU fallback, not true GDS.

In Ada job 42114353, every measured warm prefix was 100% resident before and
after timing. Cache checks and disk/NIC counters answer different questions;
logical reader bytes are not automatically physical storage bytes.

Use `payload_bytes / loop_seconds / 1e9` for useful decimal GB/s. The two recent
32-panel workloads read 33,557,176 bytes/event, so 300 events/s corresponds to
10.067 GB/s of logical payload. Do not substitute total file sizes or theoretical
PCIe/NIC limits for measured payload, device traffic or sustained bandwidth.
