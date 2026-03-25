from __future__ import annotations

from abc import ABC, abstractmethod


class GpuRuntime(ABC):
    runtime_name = None
    pipeline_name = None

    def __init__(self, run, profiler=None):
        self.run = run
        self.profiler = profiler

    def describe(self):
        return {
            'runtime': self.runtime_name,
            'pipeline': self.pipeline_name,
        }

    @abstractmethod
    def make_detector(self, name, accept_missing=False, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def handle_transition(self, rec):
        raise NotImplementedError

    @abstractmethod
    def submit_l1(self, rec):
        raise NotImplementedError

    @abstractmethod
    def pop_ready(self):
        raise NotImplementedError

    def finalize(self):
        return None
