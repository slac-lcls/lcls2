"""Resolve GPU read files from ordered SMD transitions before CPU bigdata I/O."""

from dataclasses import dataclass
import os

from psana.psexp import TransitionId
from .gpu_read_plan import ResolvedFile


@dataclass(frozen=True)
class FileEpoch:
    file: ResolvedFile
    fence: int


class GpuFileEpochs:
    """Run-scoped file state independent of mutable CPU DgramManager handles.

    An EB may include events on both sides of an Enable/chunkinfo transition.
    Replayed history can also precede its first event. Resolve a complete EB
    packet before splitting it, then give each subbatch the immutable mapping.
    """

    def __init__(self, dm):
        self._active = {
            stream: ResolvedFile(os.path.realpath(str(path)), int(dm.get_chunk_id(stream) or 0))
            for stream, path in enumerate(dm.xtc_files)
        }

    @staticmethod
    def _apply(active, service, dgrams):
        if service != TransitionId.Enable:
            return
        for stream, dg in enumerate(dgrams):
            if dg is None or not hasattr(dg, "chunkinfo"):
                continue
            entries = {
                (int(value.chunkinfo.chunkid), str(value.chunkinfo.filename))
                for value in dg.chunkinfo.values()
            }
            if not entries:
                continue
            if len(entries) != 1:
                raise ValueError(f"conflicting chunkinfo for GPU stream {stream}")
            chunk, filename = entries.pop()
            if not filename:
                raise ValueError(f"empty chunk filename for GPU stream {stream}")
            current = active[stream]
            resolved = ResolvedFile(
                os.path.realpath(os.path.join(os.path.dirname(current.path), filename)),
                chunk,
            )
            if chunk < current.chunk_id:
                continue  # step history already consumed by this BD
            if chunk == current.chunk_id and resolved != current:
                raise ValueError(f"conflicting filename for GPU stream {stream} chunk {chunk}")
            active[stream] = resolved

    def resolve(self, descriptors, transitions):
        """Return {(original event index, stream): FileEpoch}, without opening files.

        transitions contains (service, dgrams) pairs in envelope order. All
        transitions fence coalescing, but do not impose a CUDA synchronization.
        State advances through trailing transitions even in an empty GPU batch.
        """
        controls = []
        for service, dgrams in transitions:
            if not service or TransitionId.isEvent(service):
                continue
            timestamps = {int(dg.timestamp()) for dg in dgrams if dg is not None}
            if len(timestamps) != 1:
                raise ValueError("GPU transition must have one aligned timestamp")
            controls.append((timestamps.pop(), service, dgrams))
        if any(a[0] > b[0] for a, b in zip(controls, controls[1:])):
            raise ValueError("GPU transition history is not timestamp ordered")

        active = dict(self._active)
        control_index = 0
        result = {}
        for desc in sorted(descriptors, key=lambda d: (d.timestamp, d.batch_event_index, d.stream_id)):
            while control_index < len(controls) and controls[control_index][0] < desc.timestamp:
                _, service, dgrams = controls[control_index]
                self._apply(active, service, dgrams)
                control_index += 1
            if control_index < len(controls) and controls[control_index][0] == desc.timestamp:
                raise ValueError("GPU event and transition have the same timestamp")
            key = (desc.batch_event_index, desc.stream_id)
            if key in result:
                raise ValueError(f"duplicate GPU event/stream descriptor: {key}")
            result[key] = FileEpoch(active[desc.stream_id], control_index)
        for _, service, dgrams in controls[control_index:]:
            self._apply(active, service, dgrams)
        self._active = active
        return result
