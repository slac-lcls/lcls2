from contextlib import contextmanager

from common import write_json


class ReaderIoAccounting:
    def __init__(self, reader_type):
        self.reader_type = reader_type
        self.original_reset = reader_type.reset_io_stats
        self.readers = {}
        self.finished = False
        self.summary = dict(total_bytes=0, total_requests=0, requested_bytes=0,
                            useful_bytes=0, reset_windows=0, tail_windows=0,
                            positive_windows=0, readers=0, bulk_read=True,
                            compat_mode=True, io_path='CPU-fallback')

        def reset(reader):
            self.track(reader)
            snapshot = reader.io_stats()
            result = self.original_reset(reader)
            self.record(snapshot, 'reset_windows')
            return result

        reader_type.reset_io_stats = reset

    def track(self, reader):
        if reader is not None:
            self.readers[id(reader)] = reader

    def record(self, snapshot, window):
        for field in ('total_bytes', 'total_requests', 'requested_bytes', 'useful_bytes'):
            self.summary[field] += snapshot[field]
        self.summary[window] += 1
        self.summary['positive_windows'] += int(snapshot['total_bytes'] > 0)
        self.summary['bulk_read'] &= bool(snapshot['bulk_read'])
        self.summary['compat_mode'] &= bool(snapshot['compat_mode'])
        if snapshot['io_path'] != 'CPU-fallback':
            self.summary['io_path'] = 'unexpected'

    def finish(self, reader=None):
        if not self.finished:
            try:
                self.track(reader)
                for observed in self.readers.values():
                    self.record(observed.io_stats(), 'tail_windows')
                self.summary['readers'] = len(self.readers)
            finally:
                self.reader_type.reset_io_stats = self.original_reset
                self.finished = True
        return dict(self.summary)


@contextmanager
def observe_reader_io(enabled):
    accounting = None
    if enabled:
        from psana.gpu.gpu_kvikio_read import KvikioGpuReader
        accounting = ReaderIoAccounting(KvikioGpuReader)
    try:
        yield accounting
    finally:
        if accounting is not None:
            accounting.finish()


def validate_io(summary):
    if not summary['readers'] or not summary['compat_mode'] or summary['io_path'] != 'CPU-fallback':
        raise RuntimeError('GPU reader did not confirm CPU-fallback I/O')
    if summary['total_bytes'] <= 0 or summary['total_requests'] <= 0 or not summary['bulk_read']:
        raise RuntimeError('GPU reader did not perform bulk reads')


def save_io_evidence(path, accounting, reader, **evidence):
    summary = accounting.finish(reader)
    write_json(path, dict(evidence, cumulative_io=summary))
    return summary
