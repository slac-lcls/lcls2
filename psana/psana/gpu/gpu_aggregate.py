"""Keyed aggregate export for stateful tasks (PR #270 phase 1, step C).

A worker's ``end_step``/``end_run`` hands host copies of its state to one
per-BD reducer through ``ctx.emit``. The reducer merges records with equal
keys by the declared operation, learns scope completion from the framework
-- ``scope_done`` when a stream's ``end_step`` returns, whether or not it
emitted -- and calls the user's ``aggregate_sink`` inline with one merged
record per (name, key) once every stream has finished the scope. A host byte
cap turns the oldest pending scope into partial records rather than growing
without bound. A single pending key is never flushed early -- merging into it
does not grow it -- so the cap can be exceeded by at most one key's record.

What a record's flags mean, so a sink can be written against them:

``final``      no more records will follow for this key in this scope.
``truncated``  the run ended inside this step (max_events, an early close),
               or, for a run-level record, the run did not reach EndRun.
``complete``   this one record is the whole merged value of a finished step:
               final, first (``seq == 0``) and not truncated.

A key the host cap delivered early arrives as ``seq`` 0, 1, ... records that
must be added; the last carries ``final=True``, and if nothing was pending
when the step finished it is a zero-valued marker.

Delivered records stay delivered: a step every stream finished reached the
sink when it finished. What is still pending is dropped on two paths, and
the error comes out of the manager's ``close()`` both times. A failed run
(an exception in ``process`` or in a hook during the run) skips end_run and
delivers nothing pending, since a partially updated sum is not a result. A
hook that raises while the run is being finished leaves its stream's
completion unreported, so the scopes and run-level records still waiting on
that stream are never delivered either.

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
    final: bool = True
    truncated: bool = False


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
        # (name, run_instance, begin_ts, key) -> (next seq, {field: (shape, dtype)})
        # for keys delivered early by the cap, so the step's final marker can
        # carry zero-valued fields of the right shape.
        self._seq = {}
        self._bytes = 0
        self._truncated = set()  # (run_instance, begin_ts) the run ended inside
        self._run_truncated = False

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

        # Validate every field against the layout this key already has --
        # whether that layout lives in a pending entry or only in `_seq`
        # because the cap already delivered part of it -- BEFORE touching any
        # accumulator. Two reasons, both observed:
        #
        #   dtype, not just shape. np.add's default same-kind casting accepts
        #   a narrowing contribution within one kind, so an int64 partial
        #   merged into an int16 accumulator overflowed silently instead of
        #   being rejected.
        #
        #   all fields first. Validating as we merged left field 1 already
        #   modified when field 2 turned out to mismatch, and a later
        #   delivery could expose that half-updated record.
        layout, whence = None, None
        if entry is not None:
            layout = {f: (a.shape, a.dtype) for f, a in entry.fields.items()}
            whence = 'an earlier record'
        else:
            prior = self._seq.get(pkey)
            if prior is not None:
                layout = dict(prior[1])
                whence = 'a record already delivered'
        if layout is not None:
            if set(arrays) != set(layout):
                raise ValueError(
                    f'aggregate {name!r}: fields {sorted(arrays)} differ from '
                    f'{sorted(layout)} in {whence} for the same key')
            for f, a in arrays.items():
                shape, dtype = layout[f]
                if a.shape != shape:
                    raise ValueError(
                        f'aggregate {name!r}.{f}: shape {a.shape} differs from '
                        f'{shape} in {whence} for the same key')
                if a.dtype != dtype:
                    raise ValueError(
                        f'aggregate {name!r}.{f}: dtype {a.dtype} differs from '
                        f'{dtype} in {whence} for the same key')

        if entry is None:
            entry = self._pending[pkey] = _Pending(
                {f: a.copy() for f, a in arrays.items()},
                nbytes=sum(a.nbytes for a in arrays.values()))
            self._bytes += entry.nbytes
        else:
            for f, a in arrays.items():
                _OPS[spec[f]](entry.fields[f], a, out=entry.fields[f])
        entry.streams.add(stream)
        while self._bytes > self._cap and len(self._pending) > 1:
            self._flush_oldest_scope()

    # -- completion, from the framework -------------------------------------
    def scope_done(self, stream, run_instance, begin_ts, *, truncated=False):
        """One stream finished a scope. Delivers the scope's records once every
        stream has, as final; truncated if any stream reported the run ended
        inside it."""
        scope = (run_instance, begin_ts)
        if truncated:
            self._truncated.add(scope)
        done = self._done.setdefault(scope, set())
        done.add(stream)
        if len(done) >= self._n:
            self._finish_scope(scope)

    def run_done(self, stream, *, truncated=False):
        """One stream finished the run. Run-level records go out once every
        stream has, truncated if the run did not reach EndRun."""
        self._run_truncated = self._run_truncated or truncated
        self._run_done.add(stream)
        if len(self._run_done) >= self._n:
            if self._run_truncated:
                self._truncated.add((None, None))
            self._finish_scope((None, None), run_level=True)
            # Anything a scope never completed on every stream goes out
            # honestly marked partial rather than being dropped.
            self._deliver(lambda k: True, final=False)

    # -- delivery -----------------------------------------------------------
    def _flush_oldest_scope(self):
        oldest = next(iter(self._pending))
        _, run_instance, begin_ts, _ = oldest
        self._deliver(lambda k: k[1] == run_instance and k[2] == begin_ts, final=False)

    def _finish_scope(self, scope, run_level=False):
        run_instance, begin_ts = scope
        if run_level:
            select = lambda k: k[2] is None                           # noqa: E731
        else:
            select = lambda k: k[1] == run_instance and k[2] == begin_ts  # noqa: E731
        truncated = scope in self._truncated
        self._deliver(select, final=True, truncated=truncated)
        # Keys the cap delivered early and that have nothing pending now still
        # owe the sink a final marker, so it can close the key.
        for pkey in [k for k in self._seq if select(k)]:
            seq, layout = self._seq.pop(pkey)
            zeros = {f: np.zeros(shape, dtype) for f, (shape, dtype) in layout.items()}
            name, ri, ts, key = pkey
            self._sink(AggregateRecord(name, ri, ts, key, zeros, False, seq, (),
                                       final=True, truncated=truncated))
        # Nothing can arrive for a finished scope: every stream's end_step
        # has returned. Drop its bookkeeping.
        self._done.pop(scope, None)
        self._truncated.discard(scope)

    def _deliver(self, select, *, final, truncated=False):
        for pkey in [k for k in self._pending if select(k)]:
            entry = self._pending.pop(pkey)
            self._bytes -= entry.nbytes
            seq, _ = self._seq.pop(pkey, (0, None))
            name, run_instance, begin_ts, key = pkey
            complete = final and seq == 0 and not truncated
            self._sink(AggregateRecord(name, run_instance, begin_ts, key, entry.fields,
                                       complete, seq, tuple(sorted(entry.streams)),
                                       final=final, truncated=truncated))
            if not final:
                self._seq[pkey] = (seq + 1, {f: (a.shape, a.dtype)
                                             for f, a in entry.fields.items()})

    @property
    def pending_bytes(self):
        return self._bytes
