"""Verify psana modules and loaded project libraries belong to one install.

Run in fresh processes with the benchmark environment, before staging data.
Copied installs can import their own extension modules while those extensions'
embedded RPATHs silently load dependent libraries from the original prefix.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def verify_install(install):
    """Import the benchmark's native entry points and inspect actual mappings."""
    import psana
    import psana.dgram
    import psana.shmem
    import psana.eventbuilder
    import psana.smdreader

    expected = Path(install).resolve()
    origins = {
        name: str(Path(module.__file__).resolve())
        for name, module in list(sys.modules.items())
        if (name == 'psana' or name.startswith('psana.'))
        and getattr(module, '__file__', None)
    }
    wrong_modules = {name: path for name, path in origins.items()
                     if not Path(path).is_relative_to(expected)}
    if wrong_modules:
        raise RuntimeError(f'psana modules outside {expected}: {wrong_modules}')

    native_names = {path.name for path in (expected / 'lib').glob('*.so*')}
    loaded = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = Path(fields[5])
            if path.name in native_names:
                loaded.add(str(path.resolve()))
    wrong_libraries = sorted(path for path in loaded
                             if not Path(path).is_relative_to(expected))
    if wrong_libraries:
        raise RuntimeError(f'project libraries outside {expected}: {wrong_libraries}')
    required = {'libxtc.so', 'libshmemcli.so'}
    missing = required - {Path(path).name for path in loaded}
    if missing:
        raise RuntimeError(f'expected native libraries not mapped: {sorted(missing)}')

    return dict(
        rank=int(os.environ.get('SLURM_PROCID', '0')),
        install=str(expected),
        python={name: origins[name] for name in
                ('psana', 'psana.dgram', 'psana.eventbuilder', 'psana.smdreader')},
        checked_python_modules=len(origins),
        native_libraries=sorted(loaded),
        native_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                       for path in sorted(loaded)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', required=True)
    parser.add_argument('--mpi', action='store_true')
    args = parser.parse_args()
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    result = verify_install(args.install)
    if args.mpi:
        assert result['rank'] == comm.rank
        assert comm.allgather(result['rank']) == list(range(comm.size))
    print('RUNTIME_PROVENANCE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
