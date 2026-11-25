"""One-time, no-clobber import of reviewed source; never executes a campaign."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-transfer-20261115/package')
QUALIFIED = Path('/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/bundle-fixed01')

def main():
    origins = {}
    names = {
        'sim10_compute_cube_20261115': ['stages.py', 'reference.py', 'mpi_cube.py', 'contracts.py',
                                      'test_reference.py', 'test_mpi_cube.py', 'test_contracts.py'],
        'mfx_expand10_stage2_20261105': ['adapter.py', 'fixture.py', 'gpu_accept.py'],
        'mfx_expand10_stage3a_20261105': ['cube.py'],
    }
    pairs = []
    for directory, files in names.items():
        for name in files:
            pairs.append((PACKAGE/'reference/notes'/directory/name, HERE/'vendor'/directory/name))
    for name in ['runner.py']:
        pairs.append((PACKAGE/'reference/notes/sim10_compute_cube_20261115'/name, HERE/name))
    for name in ['cpu_plan.py', 'bulk_accounting.py', 'task_runtime.py', 'host_memory.py']:
        pairs.append((PACKAGE/'reference/notes/sim10_kvikio_bounce_20261113'/name, HERE/name))
    for name in ['instrument.py', 'memory_watch.py', 'affinity_probe.py', 'clean_exec.py', 'runtime.sh']:
        pairs.append((QUALIFIED/name, HERE/name))
    pairs.append((QUALIFIED/'support.py', HERE/'qualified_support.py'))
    for source, destination in pairs:
        data = source.read_bytes()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as out:
            out.write(data)
        origins[str(destination.relative_to(HERE))] = dict(origin=str(source), sha256=hashlib.sha256(data).hexdigest())
    (HERE/'source-origins.json').write_text(json.dumps(origins, indent=2)+'\n')

if __name__ == '__main__':
    main()
