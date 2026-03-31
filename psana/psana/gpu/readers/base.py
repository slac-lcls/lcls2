from __future__ import annotations

from abc import ABC, abstractmethod


class BulkReader(ABC):
    reader_name = None

    @abstractmethod
    def allocate_ingress_buffer(self, descriptors):
        raise NotImplementedError

    @abstractmethod
    def read_batch(self, descriptors, ingress_buffer):
        raise NotImplementedError

    def fill_from_batch(self, descriptor_batch):
        ingress_buffer = self.allocate_ingress_buffer(descriptor_batch.descriptors)
        self.read_batch(descriptor_batch.descriptors, ingress_buffer)
        return ingress_buffer
