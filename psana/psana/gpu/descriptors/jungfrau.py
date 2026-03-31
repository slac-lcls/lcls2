from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JungfrauPayloadDescriptor:
    timestamp: int
    service: int
    det_name: str
    file_id: int
    offset: int
    size: int
    segment_count: int
    cache_key: tuple = ()


@dataclass(frozen=True)
class JungfrauDescriptorBatch:
    timestamp: int
    service: int
    det_name: str
    descriptors: tuple[JungfrauPayloadDescriptor, ...]
    cache_key: tuple = ()

    def __len__(self):
        return len(self.descriptors)

    @property
    def is_empty(self):
        return len(self.descriptors) == 0


class JungfrauDescriptorBuilder:
    def __init__(self, run):
        self.run = run
        self._resolved_det_name = None

    def resolve_detector_name(self):
        if self._resolved_det_name is not None:
            return self._resolved_det_name

        det_classes = getattr(self.run.dsparms, "det_classes", {}) or {}
        normal = det_classes.get("normal", {})
        for (det_name, xface_name), drp_class in normal.items():
            class_name = getattr(drp_class, "__name__", "").lower()
            module_name = getattr(drp_class, "__module__", "").lower()
            if xface_name != "raw":
                continue
            if class_name.startswith("jungfrau_raw") or module_name.endswith(".jungfrau"):
                self._resolved_det_name = det_name
                return self._resolved_det_name
        return None

    def cache_key(self):
        resolved = self.resolve_detector_name() or "jungfrau"
        return ("jungfrau", getattr(self.run, "runnum", -1), resolved)

    def build_from_smd_dgrams(self, smd_dgrams, service, timestamp):
        det_name = self.resolve_detector_name()
        if det_name is None:
            return JungfrauDescriptorBatch(
                timestamp=int(timestamp),
                service=int(service),
                det_name="jungfrau",
                descriptors=(),
                cache_key=self.cache_key(),
            )

        descriptors = []
        for stream_index, dgram in enumerate(smd_dgrams):
            if dgram is None or not hasattr(dgram, "smdinfo") or not hasattr(dgram, det_name):
                continue

            offset_info = dgram.smdinfo[0].offsetAlg
            offset = int(offset_info.intOffset)
            size = int(offset_info.intDgramSize)
            if offset <= 0 or size <= 0:
                continue

            segment_count = len(getattr(dgram, det_name))
            descriptors.append(
                JungfrauPayloadDescriptor(
                    timestamp=int(timestamp),
                    service=int(service),
                    det_name=det_name,
                    file_id=int(stream_index),
                    offset=offset,
                    size=size,
                    segment_count=int(segment_count),
                    cache_key=self.cache_key(),
                )
            )
        return JungfrauDescriptorBatch(
            timestamp=int(timestamp),
            service=int(service),
            det_name=det_name,
            descriptors=tuple(descriptors),
            cache_key=self.cache_key(),
        )
