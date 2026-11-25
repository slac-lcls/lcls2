# Next task: multi-node read and staging characterization

The application scaleout campaign is closed as unsuccessful. Its 1/2/4-node measurements and runtime placement checks are in [the report](README.md). Begin a separate investigation; retain the qualified candidate and current data as controls.

1. Compare buffered POSIX IOR, the historical C++ five-file reader and psana S on GPU nodes at 1/2/4/8/16/24 nodes. Keep layer/layout comparisons on the same allocation and run them sequentially, with three repeats. Preserve actual rank/host/GPU records and per-node event/byte totals.
2. Start at 1/4 nodes with stripe counts 1/2/4/8/16/32, fixed 1 MiB size. Carry count 2 and the best alternatives upward; retest a larger-count sentinel at 16/24 nodes. Inspect actual OST union/overlap across files. Test another placement before generalizing. Tune concurrency and stripe size separately.
3. Match request ranges, CPU resources and I/O concurrency when estimating psana overhead. A 40-thread contiguous C++ read versus 16 psana workers/node is not an isolated overhead measurement. If IOR scales but shared-file reads do not, investigate layout/ranges and a controlled sharded-file variant. If matched reads scale but S does not, instrument supply, CPU, H2D and scheduling.
4. Target roughly 60 seconds after pilots; retain actual durations and interior-window limitations. The existing 400k dataset covers 100k/node only through four nodes. At 24 nodes, 2.4M events require 8.06 TB of logical JF data before overhead/copies. At the historical C++ rate, 60 seconds could require about 20 TB. Resolve data generation, cache treatment and quota before allocating that matrix. Reusing the same bytes is a separate cache/reuse experiment.
5. Interactive supports up to four nodes/four hours; propose regular QOS for 8/16/24 nodes. Prepare the concrete matrix, storage budget and node-hours before submission. Keep large data outside Git.

## Published reference methods

[NERSC architecture](https://docs.nersc.gov/systems/perlmutter/architecture/) lists **7.8 TB/s aggregate scratch reads**. Four 200 Gb/s NICs/GPU node give only a nominal 2.4 TB/s sum across 24 nodes, not a guaranteed Lustre rate. Verify actual client/LNet configuration rather than assuming all four NICs carry filesystem traffic.

The [NERSC 2021 paper](https://cug.org/proceedings/cug2021_proceedings/includes/files/pap120s2-file1.pdf), sections III/IV-B, describes buffered IOR, single-stripe 1 MiB files, one file/process, contiguous 1 MiB reads for at least 45 seconds, and client/server cache controls. Its 40.8 GB/s figure is one-OST aggregate performance.

[NERSC's 2022 acceptance slides](https://www.opensfs.org/wp-content/uploads/T2_GLockwood_measuring_io_performance.pdf), slide 30, describe 1,382 nodes/four ranks per node, file-per-process, rank shifting and 1 MiB transfers, a 90-second write with wear-out, then a 30-second read. The historical read maximum is 4,003,761 MB/s. Adapt this documented method to our client scale; pin IOR version and options, actual layout, bytes and timing. These are not the current 7.8 TB/s reproduction recipe.

Consult [Lustre tuning](https://docs.nersc.gov/performance/io/lustre/), [IOR documentation](https://ior.readthedocs.io/en/latest/) and [queue policy](https://docs.nersc.gov/jobs/policy/) at execution time. The exact public measurement protocol behind the current headline was not established. Prepare questions for NERSC if needed; do not infer a 24-node target from whole-system bandwidth. Rank shifting and separate Slurm steps alone do not prove cold caches. Client cache gates do not establish cold OSS caches. Any direct-I/O experiment must be labeled separately from the buffered application path.
