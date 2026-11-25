"""Auditable one-time edits to imported copies; original package stays intact."""
from pathlib import Path
HERE = Path(__file__).resolve().parent

def edit(name, pairs):
    p = HERE/name
    value = p.read_text()
    for before, after in pairs:
        if value.count(before) != 1:
            raise ValueError((name, before, value.count(before)))
        value = value.replace(before, after)
    p.write_text(value)

edit('runner.py', [
    ('import argparse\n', 'import argparse\nimport subprocess\n'),
    ("sys.path.extend([str(HERE.parent / 'sim10_kvikio_bounce_20261113'),\n                 str(HERE.parent / 'sim10_storage_gpu_20261114')])", "sys.path.insert(1, str(HERE/'vendor/sim10_compute_cube_20261115'))"),
    ('from support import require, write, affinity, EVENT_BYTES, histogram', 'from support import require, write, affinity, EVENT_BYTES, histogram, PACKAGE, verify_inputs'),
    ('from host_memory import snapshot, demand, validate_host_memory', "from host_memory import snapshot, demand, validate_host_memory\nimport reference\nreference.SNAPSHOT = PACKAGE/'calibration'"),
    ("rank = int(os.environ['OMPI_COMM_WORLD_RANK'])", "rank = int(os.environ['SLURM_PROCID'])"),
    ("require(len(placement['rank_uuids']) == arguments.bds + 2 and\n            len(set(placement['rank_uuids'][2:])) == arguments.bds, 'One BD per physical GPU')", "require(len(placement['rank_uuids']) == arguments.bds + 2, 'GPU rank map')\n    active_gpus = len(set(placement['rank_uuids'][2:]))\n    require(active_gpus == arguments.bds or (arguments.stage == 'S' and arguments.bds == 4 and active_gpus == 1), 'Sharing only for staging bridge')\n    granted = subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'], '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().splitlines()\n    require(set(placement['rank_uuids'][2:]) <= set(granted), 'GPU outside allocation')\n    require(placement['rank_uuids'][:2] == ['', ''], 'Helper GPU masks')"),
    ("require(comm.rank == rank and comm.size == arguments.bds + 2, 'MPI roles')", "require(comm.rank == rank and comm.size == arguments.bds + 2, 'MPI roles')\n    require('CRAY MPICH' in MPI.Get_library_version().upper(), 'Cray MPI')"),
    ("require(info['media'] == 'weka', 'This is the primary Weka cohort only')", "require(info['media'] == 'lustre', 'Qualified count2 Lustre')\n    if rank == 0:\n        verify_inputs(info)\n    from instrument import bindings, budget_peaks\n    handles_context = bindings(info['payload_records'])\n    handles = handles_context.__enter__()\n    budget_context = budget_peaks()\n    peaks = budget_context.__enter__()"),
    ("manager._n_bd_per_gpu == 1 and manager.gpu_reader.task_size", "manager._n_bd_per_gpu == arguments.bds // active_gpus and manager.gpu_reader.task_size"),
    ("conditioning = cache(info['files'], Path(info['directory']), 'weka', True)", "conditioning = cache(info['files'], Path(info['directory']), 'lustre', True)"),
    ("initial = cache(info['files'], Path(info['directory']), 'weka')", "initial = cache(info['files'], Path(info['directory']), 'lustre')"),
    ("observation['final_cache'] = cache(info['files'], Path(info['directory']), 'weka')", "observation['final_cache'] = cache(info['files'], Path(info['directory']), 'lustre')"),
    ("evidence(observation, 'weka', EVENT_BYTES", "evidence(observation, 'lustre', EVENT_BYTES"),
    ("require(io['read_size_histogram'] == (histogram(count) if count else {}), 'Read sizes')", "require(io['read_size_histogram'] == (histogram(count) if count else {}), 'Read sizes')\n        require({row['primary']['path'] for row in handles['kvikio']} == (set(info['files']) if count else set()), 'Actual KvikIO file coverage')\n        require({row['path'] for row in handles['dgram']} == set(info['files']), 'Actual DgramManager file coverage')"),
    ("row = dict(rank=rank, host=platform.node(), device=device, affinity=chosen, timestamps=delivered,", "row = dict(rank=rank, host=platform.node(), device=device, affinity=chosen, timestamps=delivered,\n               handles=handles, budget_peaks=list(peaks.values()), granted_uuids=granted, mpi=MPI.Get_library_version(),"),
    ("== arguments.bds, 'Physical placement collision')", "== active_gpus, 'Physical placement collision')\n        if arguments.kind != 'acceptance':\n            require(all(entry['timestamps'] for entry in rows[2:]), 'Full case exercises every BD')"),
    ("native_caveat='Inherited PR298 receipt has psana/meson.build mismatch; no fresh native build',", "native_caveat='Exact Perlmutter PR298 native install; job61359629 qualification reused',\n                      job=os.environ['SLURM_JOB_ID'], active_gpus=active_gpus,"),
    ("run.close_shared_memory()\n    comm.Barrier()", "run.close_shared_memory()\n    comm.Barrier()\n    handles_context.__exit__(None, None, None)\n    budget_context.__exit__(None, None, None)"),
])
edit('bulk_accounting.py', [
    ('from unittest.mock import patch', 'from unittest.mock import patch\nimport hashlib\nimport struct'),
    ('pending={}', 'pending={}\n    digests={}'),
    ('p=submit(reader,desc,ranges,total,slot,plan)', "require(len(ranges)==1 and desc.shape==(1,6), 'One datagram per read')\n        r=ranges[0]\n        timestamp,offset,size=map(int,(desc[0,2],desc[0,3],desc[0,4]))\n        require(r.file_offset==offset and r.size==size, 'Physical/logical range mismatch')\n        digests.setdefault(r.file.path,hashlib.sha256()).update(struct.pack('<QQQ',timestamp,offset,size))\n        p=submit(reader,desc,ranges,total,slot,plan)"),
    ("require(not pending,'Uncompleted input reads')", "require(not pending,'Uncompleted input reads')\n    stats['range_sha256']={key:value.hexdigest() for key,value in digests.items()}"),
])
edit('runtime.sh', [
    ('exec "$@"', 'export KVIKIO_BOUNCE_BUFFER_SIZE=16777216 PS_JF_SHARE_DERIVED=0 PS_GEO_SHARE=0\nexport SIM10_PR_SOURCE=/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/source\nexport SIM10_NATIVE_RECEIPT="$attempt/reuse-check.json"\nexec "$@"'),
])
