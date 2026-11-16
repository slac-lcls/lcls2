Review only notes/sim10_perlmutter_baseline_20261115 at 296d69985ba2cfa14e4759b035c886d0c658465f. Base 410ebba1c. No core psana changes. Actual runtime is external pinned PR298, not old core source in this checkout.
Read scope.md, PLAN.md, evidence/limitations.md, evidence/host-checks.json, evidence/host-tests.log, evidence/reuse-check.json, source-origins.json. Then inspect PM launcher/runtime/preparation, storage and input relocation, CPU/GPU mapping, runner numerical/lifetime integration, independent audit and reference probe. User explicitly approved sending this scoped code and evidence to external Claude on the latest turn; the committed blocked status predates that approval. Do not interpret that stale administrative state as missing current authorization. No compute submission is authorized.
Candidate is bounded90min bootstrap: original31GPUtests +10guards on all4GPUs; two ECC201/1003 acceptance cases; reference-cost probe; three100k S repeats for each1BD/1GPU/batch50,4BD/4GPU/batch200,4BD/1GPU/batch200 staging bridge. Four workers/BD,1stream,task1MiB,bounce16MiB,bulk512KiB,input6GiB,SMD1000,compatON,directOFF fixed. Reuse current PR298 install727files/tests86matching hashes and job61359629 qualification; selected count2stripe1MiB view. No sweep/rebuild/retransfer. Full same-node S/E/EC/ECC campaign is a later separately sized/reviewed step, not executed here.
Acceptance criteria: safe finite launcher, supported Cray MPI, live physical disjoint masks3,3,7 and3,3,7,7,7,7,helpers noCUDA,explicit granted UUID/PCI mapping,only S may share GPU; frozen calibration; local-cold owner-scoped count2 caching aftersetup; no remote-traffic claims; numerical/descriptor/lifetime/transition/drain/publication/byte/range checks preserved;4GiB physical headroom;all output retained on sharedscratch;independent exportreadback;untraced stage rates separated from reference/export;bounded phase costs;failclosed no rescue or automatic fullcampaign.
Evidence:63hosttests and64subtests pass under exact native install; AST,shellsyntax,host imports pass. Initial unactivated invocation failed imports; fixed activation. One native test relied on immediate mtime change not guaranteed on Lustre; fixture now explicitly increments mtime2sec, production reference unchanged. Added audit corruption tests and auditprocess timeouts. GPU execution, exactfullshape fit, fullreference cost and runtime results remain pending firstjobapproval. Review prelaunch readiness, not GPU execution success. MissingGPU execution is not a pass claim. Missing relevant host tests should be reported. No bundle created yet; prepare must be independently reviewed before creating it.
Read all necessary taskfiles; don't silently skip adapter code. Review originals via included vendor/origin evidence rather than outside paths that restricted mode cannot access. Do not run commands, edit, delegate or start another review loop. Return stable PM-R1-xx findings and exact reviewedSHA. Any unavailable required evidence is a limitation, never a pass.

ROUND 2: Re-review fixes to all PM-R1-01..08 and detect regressions. Read review/round01.json and review/response-01.md. User authorization persists (AUTHORIZATION.md). Do not run commands or launch another reviewer. Prior verdict changes_requested, not pass. Additional coordinator bug (missing expand oracle) is fixed and needs review. The reference probe now precedes acceptance. Manifest pin plus array hashing, complete adapter-port.diff with calibration source SHA gate, UTC scontrol and remaining-time bounds, UUID/PCI runtime and audit checks, lock failure status and timeout diagnostics are all changed. Tiny-file ECC tests use a helper shape override only in tests; production audit calls the fixed full-shape default. Guard import tests are not GPU execution.
The committed host evidence is the precommit run. Independently reran the exact current commit from the coordinator worktree with the task tree clean: 90 passed +64 subtests; AST, bash syntax, installed psana prefix and imports pass. Full raw receipt follows and includes all bundled file types. This run is after the commit (therefore cannot be committed inside the same SHA). Host output at host-checks06, and no test changed source. Do not claim to have run it yourself.

EXACT COMMIT HOST RECEIPT:
{
  "git_head": "296d69985ba2cfa14e4759b035c886d0c658465f",
  "task_working_status": "",
  "status": "pass",
  "exitcode": 0,
  "command": [
    "/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env/bin/python",
    "-m",
    "pytest",
    "-q",
    "--basetemp=/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115/host-checks06/pytest-tmp",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/test_port.py",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/test_audit_port.py",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/test_boundaries.py",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/vendor/sim10_compute_cube_20261115/test_reference.py",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/vendor/sim10_compute_cube_20261115/test_mpi_cube.py",
    "/global/u2/m/monarin/lcls2_worktree/features/psana2-gpu/notes/sim10_perlmutter_baseline_20261115/vendor/sim10_compute_cube_20261115/test_contracts.py"
  ],
  "psana": "/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/source/install_psana/lib/python3.11/site-packages/psana/__init__.py",
  "source_hashes": {
    "runtime.py": "1fcf367da91bbe3a84d427da91f56a50f53330decc95dfe2307305d432832a2a",
    "plan.py": "5dffaa6e144ced8629a18e6e6ac3e5d8931eccf610c427e715a602ee54e8fa3d",
    "check_tests.py": "47daa0449e19653e331efa5bac7923b5489e556e49f202920e1185b1dcea8901",
    "port_sources.py": "7b068ba6e61cc12187953856bd8ea2d7cee894c24ad8362fef2ee11129de1025",
    "import_sources.py": "9ed995fd32522c7e3a9a202141ad1eca1a6d83a71ddbe3f0f59c3b72cb2f2999",
    "check_reuse.py": "d7a254e1e88712b925c4822e01ff571359e126c6ec392b493c5ddda4c82b7265",
    "qualified_support.py": "c92d46a65d2610adca284be5589aeffd2d51e2da482be9a6d4fcf83f1a51279f",
    "bulk_accounting.py": "7be4ad8453a27c2575000e37873f57e03c9654b5289fe9cdedf216da8060f393",
    "prepare.py": "323d00f73ac64186e502e3feca15a9781cb01290856a718e54ec43759c42018e",
    "cpu_plan.py": "fa6a756b67c1e29371d195e89e08264d49268374d0b270ecbf3de5fbfce510be",
    "device_check.py": "034065fc18c2a829be9d4fefb07f0446f889496883413df97f770c7eceeb0c1b",
    "runtime.sh": "988ae2532369f109ba6548f4faf787ba621c4d5177f862bc1301c21bd1b80aa7",
    "source-origins.json": "c10368dd283b01d024c8114069ef74b310e01e69a69ee035b86f0ff2f31522c4",
    "campaign.py": "7fbc13d28c84394c2cb26ca56e9b3412db5126d08ca2f93f086c0689e1e43396",
    "test_audit_port.py": "bcc478f45da8e455f211b42fde3a30a4977c43994793d0df1d52169bbe833830",
    "memory_watch.py": "4917b0894dbd964cc07397023bab4b9b681961d6b56af752342578d24094c597",
    "expected-native.json": "6bf2c4b6e5e75100c99ccba221f81426d69e039407f330bd970a7af31ff1f816",
    "support.py": "96679115bea5bf08d58fd15228d632e7ed6a2236b32e0973f6c69a20f61be188",
    "affinity_probe.py": "dd0e35a36610cfce1d488cc7f9f56c977e6aa74dfe164dd6afad6078496e3732",
    "test_port.py": "9d135b16e281e71ea9ff427226c32c64d74e8330c2ea8a5c7ac32df4c620c870",
    "storage.py": "0a4c77f99ba97b4bfd20943a6e538d140129d76a90fea2d6d6d5cfca0d029070",
    "host_memory.py": "e4142f35478934c917adf7c15739e840ade9764f0722b7233f1c9e51b0927c8d",
    "task_runtime.py": "0157be63682cbcefebd608e0faf01b0d9fde69a8ef3b78eb993dc06b194b3a03",
    "run.sbatch": "99167059e77f4cf17c49a34bedd6eccdd7337cb65d5e0ee39296a9f997527270",
    "host_checks.py": "b98ed8919c26cc80b09cd3f7891af269910a1e9402d29da92d1ec67d19a734a0",
    "clean_exec.py": "4994a609b58950d7072d4c983bda332ea65644cbf30c6e3ad6022e009340d492",
    "audit.py": "71995f80d6d88365b05b39938cc140a590cf1eecc2cd085fa67c836731e4cdff",
    "reference_probe.py": "4b00ff774fe8bcef65cdc1d6fd4641e88f281cc5dfd7a57c8ca368d4a0629a42",
    "test_boundaries.py": "44b2f77b752f985fc71e254ebf3fd3f5fee9b8e7f0068e6c441ff9262c7f98b9",
    "instrument.py": "aa7d6d720bf1bd1d7a8349db3dff8601820ba6d56042bc9a3d80f3cc17f23859",
    "runner.py": "f7e80ba189ad0edecc520e9d7b0723902774e4b9a609a188fc6c5cf87a6f71bb",
    "status.json": "db50cb056722a5c466c38bba14260f61574d7e593e46df3219a0463d8c92782b",
    "vendor/sim10_compute_cube_20261115/test_mpi_cube.py": "dddfef187b69f588afcff01284261b194cf380d5e5aef98e6bf2ea1961471f92",
    "vendor/sim10_compute_cube_20261115/stages.py": "b19ec2131eeca45189af25a5e213139277527463db779ad1e7f6f6f78cad5f69",
    "vendor/sim10_compute_cube_20261115/mpi_cube.py": "15fe0d105a3fa16c0db2a0130789281568af8e3f0f0b171ea6229e38a2cec4c0",
    "vendor/sim10_compute_cube_20261115/test_contracts.py": "c683f55a4404a1605b2b7d0349cda73eacc2d6152151f8ebf1127f7954886812",
    "vendor/sim10_compute_cube_20261115/reference.py": "d161a4940c179c7d8f982318323b1d035b43078dec603703a024e392beda213e",
    "vendor/sim10_compute_cube_20261115/contracts.py": "3d1ade042cf0c0970917ab85f36174b199f66719c66e685b5b4497106e25b856",
    "vendor/sim10_compute_cube_20261115/test_reference.py": "15a8bee8ac37183ac0188739ba4af824c7a19f5d18ce39166fe23552061d18bd",
    "vendor/mfx_expand10_stage2_20261105/adapter.py": "fe6fbc73ec3d3198955342d59af4d88a2a1811af48d882fe6a3986c7b28a993c",
    "vendor/mfx_expand10_stage2_20261105/gpu_accept.py": "7ebc741c643a0b832db03308f9d4122bb962127e6ab5c5fcfbe7f6b6f2078d4d",
    "vendor/mfx_expand10_stage2_20261105/verify.py": "b391ffd32b229963c16ee50c0eff0a7f7ec7a0bca39afeec805c0b58c6a192a3",
    "vendor/mfx_expand10_stage2_20261105/fixture.py": "7e62e50669bb6a3315bbdf7be1f1ae471d76b36973f3074e1dd86806db62a8a4",
    "vendor/mfx_expand10_stage3a_20261105/cube.py": "3b3452da6d05edb8b9b3d7ad23efd4f18d210ac34ebd0849ebf5099cabdf4415"
  },
  "limitations": [
    "CPU-only; no GPU acceptance or scheduler execution. Host checks ran outside the unavailable Codex sandbox."
  ]
}
PYTEST SUMMARY:
90 passed, 221 warnings, 64 subtests passed in 4.63s
