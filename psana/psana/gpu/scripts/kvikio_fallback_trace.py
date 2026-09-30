"""Benchmark-only native fallback trace coordinator (KvikIO 24.08.02)."""
import ctypes
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import time
import threading


class FallbackTrace:
    def __init__(self, library):
        import kvikio
        import kvikio._lib.libkvikio as extension
        from psana.gpu.gpu_kvikio_read import KvikioGpuReader
        assert kvikio.__version__ == '24.08.02', kvikio.__version__
        self.lib = ctypes.CDLL(str(library))
        self.lib.fallback_install.argtypes = [ctypes.c_char_p]
        self.lib.fallback_dump.argtypes = [ctypes.c_char_p]
        self.lib.fallback_batch.argtypes = [ctypes.c_uint]
        for name in ('fallback_begin', 'fallback_end', 'fallback_batch'):
            getattr(self.lib, name).restype = None
        assert self.lib.fallback_install(os.fsencode(extension.__file__)) == 0, 'CUDA shim ABI mismatch'
        self.provenance = dict(kvikio_version=kvikio.__version__, extension=extension.__file__,
            caller_tid=threading.get_native_id(), gds_threshold=kvikio.defaults.gds_threshold(),
            task_size=kvikio.defaults.task_size(),
            extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
            library=str(library), library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest())
        self.active = False
        self.batches = []
        self.pending = {}
        self.fd_paths = {}
        self.paths_seen = set()
        issue_name = '_submit_read' if hasattr(KvikioGpuReader, '_submit_read') else 'issue_batch'
        issue, wait = getattr(KvikioGpuReader, issue_name), KvikioGpuReader.wait_batch

        @wraps(issue)
        def issue_trace(reader, *args, **kwargs):
            if not self.active:
                return issue(reader, *args, **kwargs)
            number = len(self.batches) + 1
            self.lib.fallback_batch(number)
            begin = time.perf_counter_ns()
            result = issue(reader, *args, **kwargs)
            end = time.perf_counter_ns()
            paths = {r.file.path for r, _, _ in result.futures}
            if paths - self.paths_seen:
                # KvikIO can hold both direct and buffered handles. Capture
                # their identities while open; the benchmark uses stable files.
                for fd in Path('/proc/self/fd').iterdir():
                    try:
                        path = os.readlink(fd)
                    except FileNotFoundError:
                        continue
                    if path in paths:
                        number_fd = int(fd.name)
                        assert self.fd_paths.get(number_fd, path) == path
                        self.fd_paths[number_fd] = path
                self.paths_seen.update(paths)
            entry = dict(batch=number, issue_begin=begin, issued_ns=result.issued_ns,
                         issue_end=end, requests=len(result.futures),
                         requested_bytes=sum(size for _, size, _ in result.futures),
                         ranges=[dict(file=r.file.path, offset=r.file_offset, size=size)
                                 for r, size, _ in result.futures])
            self.batches.append(entry)
            self.pending[id(result)] = entry
            return result

        @wraps(wait)
        def wait_trace(reader, pending, *args, **kwargs):
            entry = self.pending.get(id(pending)) if self.active else None
            if entry is None:
                return wait(reader, pending, *args, **kwargs)
            entry['wait_begin'] = time.perf_counter_ns()
            result = wait(reader, pending, *args, **kwargs)
            entry['wait_end'] = time.perf_counter_ns()
            del self.pending[id(pending)]
            return result

        setattr(KvikioGpuReader, issue_name, issue_trace)
        KvikioGpuReader.wait_batch = wait_trace

    def begin(self):
        self.batches.clear()
        self.pending.clear()
        self.fd_paths.clear()
        self.paths_seen.clear()
        self.lib.fallback_begin()
        self.active = True

    def end(self, path):
        self.active = False
        self.lib.fallback_end()
        assert not self.pending, 'undrained reads'
        path = Path(path)
        n = self.lib.fallback_dump(os.fsencode(path))
        assert n > 0, f'empty/overflow/failed trace: {n}'
        result = dict(self.provenance, records=n, record_size=64, binary=str(path),
                      clock='CLOCK_MONOTONIC nanoseconds', batches=self.batches,
                      attribution='file_ranges', fd_paths=self.fd_paths)
        path.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
        return dict(binary=str(path), metadata=str(path.with_suffix('.json')), records=n)
