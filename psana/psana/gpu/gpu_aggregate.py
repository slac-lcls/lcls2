"""Keyed aggregate export for stateful tasks (PR #270 phase 1, step C).

A worker's ``end_step``/``end_run`` hands host copies of its state to one
per-BD reducer through ``ctx.emit``. The reducer merges records with equal
keys by the declared operation, learns scope completion from the framework
-- ``scope_done`` when a stream's ``end_step`` returns, whether or not it
emitted -- and calls the user's ``aggregate_sink`` inline with one merged
record per (name, key) once every stream has finished the scope. A host byte
cap turns the oldest pending scope into partial records rather than growing
without bound; partials for one key carry increasing ``seq`` and must be
added by the sink.

Host-only: NumPy in, NumPy out. No CUDA, no MPI, no collective.
"""
from dataclasses import dataclass, field

import numpy as np

_OPS = {'sum': np.add}


def normalise_aggregate_outputs(outputs):
    """``{name: {field: op}}`` -> hashable ``((name, ((field, op), ...)), ...)``."""
    if not outputs:
        return ()
    if not isinstance(outputs, dict):
        outputs = dict(outputs)
    result = []
    for name, fields in outputs.items():
        if not isinstance(name, str) or not name:
            raise ValueError('aggregate output names must be nonempty strings')
        fields = dict(fields)
        if not fields:
            raise ValueError(f'aggregate output {name!r} declares no fields')
        for fname, op in fields.items():
            if not isinstance(fname, str) or not fname:
                raise ValueError(f'aggregate output {name!r}: field names must be nonempty strings')
            if op not in _OPS:
                raise ValueError(f'aggregate output {name!r}.{fname}: unsupported op {op!r}; '
                                 f'supported: {sorted(_OPS)}')
        result.append((name, tuple(fields.items())))
    return tuple(result)


@dataclass
class AggregateRecord:
    """What the sink receives: one key of one output, merged over the streams
    that reported it. ``begin_ts`` is None for a run-level record."""
    name: str
    run_instance: object
    begin_ts: object
    key: object
    fields: dict
    complete: bool
    seq: int
    streams: tuple


@dataclass
class _Pending:
    fields: dict
    streams: set = field(default_factory=set)
    nbytes: int = 0


class AggregateReducer:
    def __init__(self, outputs, n_streams, sink, host_bytes):
        self._spec = {name: dict(fields) for name, fields in outputs}
        self._n = int(n_streams)
        self._sink = sink
        self._cap = int(host_bytes)
        self._pending = {}       # (name, run_instance, begin_ts, key) -> _Pending
        self._done = {}          # (run_instance, begin_ts) -> set(stream)
        self._run_done = set()
        self._seq = {}           # (name, run_instance, begin_ts, key) -> next seq
        self._bytes = 0

    # -- data ---------------------------------------------------------------
    def add(self, name, run_instance, begin_ts, key, fields, stream):
        spec = self._spec.get(name)
        if spec is None:
            raise KeyError(f'{name!r} is not a declared aggregate output')
        if set(fields) != set(spec):
            raise ValueError(f'aggregate {name!r} expects fields {sorted(spec)}, got {sorted(fields)}')
        pkey = (name, run_instance, begin_ts, key)
        entry = self._pending.get(pkey)
        arrays = {f: np.asarray(v) for f, v in fields.items()}
        if entry is None:
            entry = self._pending[pkey] = _Pending(
                {f: a.copy() for f, a in arrays.items()},
                nbytes=sum(a.nbytes for a in arrays.values()))
            self._bytes += entry.nbytes
        else:
            for f, a in arrays.items():
                have = entry.fields[f]
                if have.shape != a.shape:
                    raise ValueError(f'aggregate {name!r}.{f}: shape {a.shape} differs from '
                                     f'an earlier record {have.shape} for the same key')
                _OPS[spec[f]](have, a, out=have)
        entry.streams.add(stream)
        while self._bytes > self._cap and len(self._pending) > 1:
            self._flush_oldest_scope(complete=False)

    # -- completion, from the framework -------------------------------------
    def scope_done(self, stream, run_instance, begin_ts):
        done = self._done.setdefault((run_instance, begin_ts), set())
        done.add(stream)
        if len(done) >= self._n:
            self._flush(lambda k: k[1] == run_instance and k[2] == begin_ts, complete=True)

    def run_done(self, stream):
        self._run_done.add(stream)
        if len(self._run_done) >= self._n:
            self._flush(lambda k: k[2] is None, complete=True)
            # Anything a scope never completed on every stream goes out
            # honestly marked partial rather than being dropped.
            self._flush(lambda k: True, complete=False)

    # -- delivery -----------------------------------------------------------
    def _flush_oldest_scope(self, complete):
        oldest = next(iter(self._pending))
        _, run_instance, begin_ts, _ = oldest
        self._flush(lambda k: k[1] == run_instance and k[2] == begin_ts, complete=complete)

    def _flush(self, select, complete):
        for pkey in [k for k in self._pending if select(k)]:
            entry = self._pending.pop(pkey)
            self._bytes -= entry.nbytes
            seq = self._seq.get(pkey, 0)
            self._seq[pkey] = seq + 1
            name, run_instance, begin_ts, key = pkey
            self._sink(AggregateRecord(name, run_instance, begin_ts, key, entry.fields,
                                       complete, seq, tuple(sorted(entry.streams))))

    @property
    def pending_bytes(self):
        return self._bytes
