"""
detector_cache.py

Utility class for caching selected attributes of CalibConstants (_calibc_)
from psana Detector objects to avoid redundant and expensive computation
(e.g., pixel coordinate arrays, geometry info, and derived image helpers).

Designed for use in LCLS2 environments where psana.Detector is initialized
repeatedly in live or batch modes, but pixel geometry does not change across runs.

Usage:
    from psana.detector.detector_cache import DetectorCacheManager

    cache_mgr = DetectorCacheManager(det.raw)
    cache_mgr.ensure()
"""

import pickle
from pathlib import Path

from psana.utils import Logger


class DetectorCacheManager:
    """
    Manages caching of expensive-to-compute attributes from CalibConstants
    (e.g., pixel coordinate indexes and geometry helpers) to disk using pickle.

    Attributes are restored when possible to speed up detector setup.
    """

    def __init__(self, det, check_before_update=False, cache_dir="/dev/shm", logger=None):
        """
        Initialize the cache manager with the detector and cache directory.

        Parameters:
        det (psana.Detector): The detector object (e.g. `det.raw`)
        cache_dir (str): Directory where pickle files will be stored
        """
        self.det = det
        self.cc = det._calibc_
        self.check_before_update = check_before_update
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / f"{det._det_name}_calibc_cache.pkl"
        if logger is None:
            self.logger = Logger(name="DetectorCacheManager")
        else:
            self.logger = logger

        # List of CalibConstants attributes that are safe to serialize
        self.attrs_to_cache = [
            '_rc_tot_max', '_pix_rc', 'img_entries',
            'dmulti_pix_to_img_idx', 'dmulti_imgidx_numentries',
            '_interpol_pars', 'img_pix_ascend_ind', 'img_holes',
            'hole_rows', 'hole_cols', 'hole_inds1d'
        ]

    def save(self):
        """
        Saves selected attributes of CalibConstants to a pickle file.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_data = {
            attr: getattr(self.cc, attr)
            for attr in self.attrs_to_cache
            if hasattr(self.cc, attr)
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        self.logger.info(f"Saved cache to {self.cache_file}")

    def ensure(self):
        """
        Ensures the cached data is available. Loads from file if possible;
        otherwise triggers computation via one event and then caches it.

        This method assumes the detector has an attached `.run` as `det._run`.
        """
        needs_update = False
        if self.check_before_update:
            if not self.cache_file.exists():
                needs_update = True
        else:
            needs_update = True

        if needs_update:
            self.logger.info("Computing and caching CalibConstants attributes...")
            try:
                evt = next(self.det._run.events())
                _ = self.det.image(evt)  # triggers CalibConstants initialization
                self.cc = self.det._calibc_  # refresh reference in case it was updated
                self.save()
            except Exception:
                import traceback
                self.logger.warning("Error while computing attributes:")
                self.logger.warning(traceback.format_exc())

    def clear(self):
        """
        Deletes the existing cache file, if it exists.
        """
        if self.cache_file.exists():
            self.cache_file.unlink()
            self.logger.info(f"Cleared cache at {self.cache_file}")
