"""Freeze installed KvikIO identities without importing CUDA libraries."""
import hashlib
from pathlib import Path
import sysconfig


def kvikio_identity():
    root=Path(sysconfig.get_path('purelib'))
    extension=list((root/'kvikio/_lib').glob('file_handle*.so'))
    if len(extension)!=1:
        raise ValueError('Expected exactly one installed KvikIO file_handle extension')
    files=extension+[root/p for p in (
        'kvikio/cufile.py','kvikio/defaults.py','libkvikio/GIT_COMMIT',
        'libkvikio/lib64/libkvikio.so',
        'libkvikio/include/kvikio/detail/parallel_operation.hpp',
        'libkvikio/include/kvikio/bounce_buffer.hpp',
        'libkvikio/include/kvikio/detail/posix_io.hpp')]
    return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
