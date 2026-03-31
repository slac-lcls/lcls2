from __future__ import annotations

from abc import ABC, abstractmethod


class IngressParser(ABC):
    parser_name = None

    @abstractmethod
    def parse_batch(self, descriptor_batch, ingress_buffer):
        raise NotImplementedError
