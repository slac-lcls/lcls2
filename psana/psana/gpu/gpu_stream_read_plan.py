"""CPU-only Stage 1 planner for bounded reads interleaved across streams.

One invocation covers one EB batch. This is a request/ownership plan, not an
allocation or an executable schedule: the future scheduler must honor both
group dependencies and the shared budget before submitting any request.
"""
from dataclasses import dataclass, replace
from typing import Optional, Tuple

from .gpu_read_plan import ResolvedDgram, ResolvedFile, _uint64, build_read_plan


@dataclass(frozen=True)
class StreamReadGroup:
    group_id: int
    stream_id: int
    fence_id: int
    file: ResolvedFile
    file_offset: int
    size: int
    dgrams: Tuple[ResolvedDgram, ...]
    small: bool
    # A later small group must wait for this group's last consumer, not its I/O.
    after_group: Optional[int] = None

    @property
    def first_event(self):
        return self.dgrams[0].batch_event_index

    @property
    def event_stop(self):
        return self.dgrams[-1].batch_event_index + 1


@dataclass(frozen=True)
class StreamReadPlan:
    batch_id: int
    n_events: int
    groups: Tuple[StreamReadGroup, ...]
    empty_dgrams: Tuple[ResolvedDgram, ...]
    useful_bytes: int
    input_capacity_bytes: int
    # Conservative capacity for one largest group from every physical stream.
    # This is a progress bound, NOT a measured/runtime peak or total GPU quota.
    one_group_per_stream_bytes: int
    small_target_bytes: int


def build_stream_read_plan(descriptors, *, n_events, batch_id=0,
                           small_target_bytes=1 << 20,
                           input_capacity_bytes, fence_by_event=None):
    """Plan exact, adjacent-only reads with independent group-relative offsets.

    Dgrams below the target coalesce up to it. Dgrams at/above the target are
    single-dgram requests (large dgrams may exceed the target). Groups are
    offered in (first event, stream ID) order, never file-major order. Missing
    stream events need no synthetic row; zero-size descriptors remain metadata.

    ``after_group`` describes one outstanding small group per physical stream
    within this batch; the runtime must carry that credit across batches.
    A scheduler must skip blocked streams rather than wait on the entire list.
    Large-group concurrency and parser/kernel batch sizes are independent and
    will be set by runtime byte admission; this planner does not serialize them.

    input_capacity_bytes is the caller's RAW INPUT allowance after reserving
    parser, detector, output, cache and margin bytes. Reject plans whose
    conservative one-group-per-stream progress bound exceeds that allowance.
    This does not reserve space for all requests or permit issuing them all.
    fence_by_event supplies a monotonically increasing transition epoch for
    every represented event. Omit only for a known transition-free batch.
    """
    n_events = _uint64('n_events', n_events)
    batch_id = _uint64('batch_id', batch_id)
    target = _uint64('small_target_bytes', small_target_bytes)
    capacity = _uint64('input_capacity_bytes', input_capacity_bytes)
    if not target:
        raise ValueError('small_target_bytes must be positive')
    descriptors = tuple(descriptors)
    if any(not isinstance(d, ResolvedDgram) for d in descriptors):
        raise TypeError('descriptors must contain ResolvedDgram records')
    total = sum(d.size for d in descriptors)
    # Reuse the existing duplicate, timestamp, overlap and uint64 validation.
    build_read_plan(descriptors, capacity_bytes=total, batch_id=batch_id)
    events = sorted({d.batch_event_index for d in descriptors})
    if events and events[-1] >= n_events:
        raise ValueError('descriptor event is outside this batch')
    if fence_by_event is None:
        fences = {e: 0 for e in events}
    else:
        missing = [e for e in events if e not in fence_by_event]
        if missing:
            raise ValueError(f'missing transition fence for event {missing[0]}')
        fences = {e: _uint64('fence_id', fence_by_event[e]) for e in events}
    if any(fences[a] > fences[b] for a, b in zip(events, events[1:])):
        raise ValueError('transition fences must not move backwards')
    groups = []
    for d in sorted((d for d in descriptors if d.size),
                    key=lambda d: (d.stream_id, d.batch_event_index)):
        small = d.size < target
        previous = groups[-1] if groups else None
        if (small and previous is not None and previous.small
                and previous.stream_id == d.stream_id
                and previous.file == d.file and previous.fence_id == fences[d.batch_event_index]
                and previous.file_offset + previous.size == d.file_offset
                and previous.size + d.size <= target):
            groups[-1] = replace(previous, size=previous.size+d.size,
                                 dgrams=previous.dgrams+(d,))
        else:
            groups.append(StreamReadGroup(-1, d.stream_id, fences[d.batch_event_index],
                                          d.file, d.file_offset, d.size, (d,), small))
    ordered = sorted(groups, key=lambda g: (g.first_event, g.stream_id))
    latest_small, maxima, result = {}, {}, []
    for index, g in enumerate(ordered):
        result.append(replace(g, group_id=index,
                              after_group=latest_small.get(g.stream_id) if g.small else None))
        if g.small:
            latest_small[g.stream_id] = index
        maxima[g.stream_id] = max(maxima.get(g.stream_id, 0), g.size)
    progress = sum(maxima.values())
    if progress > capacity:
        raise ValueError(f'one-group-per-stream input requires {progress} bytes; '
                         f'input_capacity_bytes={capacity}; reduce coalescing target or reserve more input')
    return StreamReadPlan(batch_id, n_events, tuple(result),
                          tuple(d for d in descriptors if not d.size), total,
                          capacity, progress, target)
