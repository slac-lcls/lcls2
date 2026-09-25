"""Native trace coordination must observe direct groups through shared submit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import sys

from psana.gpu.gpu_kvikio_read import KvikioGpuReader
from test_gpu_bulk_read import io  # noqa: F401
from test_gpu_direct_group_read import group


def test_native_trace_observes_direct_group_issue_and_wait(io, monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[3] / 'gpu/scripts/kvikio_fallback_trace.py'
    spec = importlib.util.spec_from_file_location('direct_group_trace', source)
    trace_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace_module)
    library = tmp_path / 'mock.so'
    library.write_bytes(b'fake native library identity')
    extension = NS(__file__=str(library))
    backend = sys.modules['kvikio']
    monkeypatch.setattr(backend, '__version__', '24.08.02', raising=False)
    monkeypatch.setattr(backend, '_lib', NS(libkvikio=extension), raising=False)
    monkeypatch.setitem(sys.modules, 'kvikio._lib', backend._lib)
    monkeypatch.setitem(sys.modules, 'kvikio._lib.libkvikio', extension)
    monkeypatch.setattr(backend, 'defaults', NS(gds_threshold=lambda: 0, task_size=lambda: 4), raising=False)
    lib = NS(**{name: (lambda *args: 0) for name in (
        'fallback_install', 'fallback_dump', 'fallback_batch', 'fallback_begin', 'fallback_end')})
    monkeypatch.setattr(trace_module.ctypes, 'CDLL', lambda _: lib)
    # Ensure pytest restores methods replaced inside the coordinator.
    monkeypatch.setattr(KvikioGpuReader, '_submit_read', KvikioGpuReader._submit_read)
    monkeypatch.setattr(KvikioGpuReader, 'wait_batch', KvikioGpuReader.wait_batch)
    trace = trace_module.FallbackTrace(library)
    trace.begin()
    io.files = {'/data': bytes(range(64))}
    reader = KvikioGpuReader()
    pending = reader.issue_group(group(), slot_id=0)
    assert len(trace.batches) == 1
    entry = trace.batches[0]
    assert entry['requests'] == 1 and entry['requested_bytes'] == 12
    assert entry['ranges'] == [dict(file='/data', offset=5, size=12)]
    assert id(pending) in trace.pending
    reader.wait_batch(pending)
    assert not trace.pending
    assert entry['issued_ns'] <= entry['issue_end'] <= entry['wait_begin'] <= entry['wait_end']
    reader.close()
