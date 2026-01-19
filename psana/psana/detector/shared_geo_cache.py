"""
SharedGeoCache
==============

Draft helper for storing geometry-derived detector arrays in MPI shared memory.

This class is intentionally minimal; integration points in AreaDetector and
CalibConstants should call into it to get or build shared arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from psana.utils import Logger


@dataclass(frozen=True)
class SharedGeoKey:
    """Immutable key identifying one detector geometry cache group."""
    det_name: str
    drp_class: str
    geom_id: str


class SharedGeoCache:
    """Thin wrapper around a MPISharedMemory-like object."""

    def __init__(self, shared_mem: Any = None, logger: Optional[Logger] = None):
        self.shared_mem = shared_mem
        self.logger = logger or Logger(name="SharedGeoCache")
        self._local_meta: Dict[str, Dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self.shared_mem is not None

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^0-9A-Za-z_]+", "_", name)

    @staticmethod
    def _stable_json(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    @classmethod
    def build_geom_id(
        cls,
        geotxt: Optional[str],
        segnums: Optional[Iterable[int]],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a short hash for geometry-dependent cache invalidation."""
        payload = {
            "geotxt": geotxt or "",
            "segnums": list(segnums) if segnums is not None else None,
            "kwargs": kwargs or {},
        }
        digest = hashlib.sha1(cls._stable_json(payload).encode("utf-8")).hexdigest()
        return digest[:12]

    @classmethod
    def make_key(cls, det_name: str, drp_class: str, geom_id: str) -> SharedGeoKey:
        return SharedGeoKey(
            det_name=cls._safe_name(det_name),
            drp_class=cls._safe_name(drp_class),
            geom_id=cls._safe_name(geom_id),
        )

    @staticmethod
    def _prefix(key: SharedGeoKey) -> str:
        return f"geo_{key.det_name}_{key.drp_class}_{key.geom_id}"

    def record_meta(self, key: SharedGeoKey, meta: Dict[str, Any]) -> None:
        """Record small metadata locally (not shared)."""
        self._local_meta[self._prefix(key)] = meta

    def get_meta(self, key: SharedGeoKey) -> Optional[Dict[str, Any]]:
        return self._local_meta.get(self._prefix(key))

    def get_or_allocate(
        self,
        key: SharedGeoKey,
        name: str,
        shape: Iterable[int],
        dtype: Any,
        zero_init: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Return a shared array, allocating if needed.

        Returns:
            (array, created)
        """
        if not self.enabled:
            raise RuntimeError("SharedGeoCache is not enabled (no shared_mem).")

        dtype = np.dtype(dtype)
        shape_tuple = tuple(int(dim) for dim in shape)
        full_name = f"{self._prefix(key)}_{name}"

        if self.shared_mem.has_array(full_name):
            handle = (
                self.shared_mem.get_handle(full_name)
                if hasattr(self.shared_mem, "get_handle")
                else None
            )
            if handle and (handle.shape != shape_tuple or handle.dtype != dtype):
                raise ValueError(
                    f"Shared array {full_name} shape/dtype mismatch: "
                    f"{handle.shape}/{handle.dtype} vs {shape_tuple}/{dtype}"
                )
            return self.shared_mem.get_array(full_name), False

        array = self.shared_mem.allocate_array(
            full_name, shape_tuple, dtype, zero_init=zero_init
        )
        return array, True

    def barrier(self) -> None:
        if self.enabled and hasattr(self.shared_mem, "barrier"):
            self.shared_mem.barrier()
