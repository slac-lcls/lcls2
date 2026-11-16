# Scope: psana-GPU stripe screening plan

User requests a reviewed parameter list before running psana2-gpu read tests,
using the previous 512KiB/KvikIO/batch settings and the build from PR298.
This turn prepares and independently reviews a concrete plan only. No build,
GPU tests, data writes, cache eviction or Slurm benchmark submission yet.
Read-only discovery, fetching the pinned PR source, and local documentation/
review commits are authorized. External Claude evidence review was explicitly
approved earlier in this session and is requested again through the review loop.
Default limit: three plan review rounds. Existing stripe results remain intact.

Primary experiment is framework GPU input/descriptor staging on the accepted
100k five-JF-file fixture. It is not the cube, a 400k corpus, GDS qualification,
or multi-node scaling. Those need separate matched workload/placement tests.
Execution will use lcls_g/premium after the user has seen this parameter list.
