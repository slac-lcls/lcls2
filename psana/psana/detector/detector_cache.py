"""
detector_cache.py

Utility class for caching selected attributes of CalibConstants (_calibc_)
from psana Detector objects to avoid redundant and expensive computation
(e.g., pixel coordinate arrays, geometry info, and derived image helpers).

Designed for use in LCLS2 environments where psana.Detector is initialized
repeatedly in live or batch modes, but pixel geometry does not change across runs.

Usage:
    from psana.detector.detector_cache import DetectorCacheManager

    cache_mgr = DetectorCacheManager(det)
    cache_mgr.ensure()
"""

import pickle
import time
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
        det (psana.Detector): The detector container object (not det.raw)
        cache_dir (str): Directory where pickle files will be stored
        """
        self.det = det
        self.det_name = getattr(det, "_det_name", "unknown")
        self.check_before_update = check_before_update
        self.cache_dir = Path(cache_dir)
        self.cache_file = self._get_cache_file_path(self.det_name, self.cache_dir)
        if logger is None:
            self.logger = Logger(name="DetectorCacheManager")
        else:
            self.logger = logger

        self.attrs_to_cache = [
            '_rc_tot_max', '_pix_rc', 'img_entries',
            'dmulti_pix_to_img_idx', 'dmulti_imgidx_numentries',
            '_interpol_pars', 'img_pix_ascend_ind', 'img_holes',
            'hole_rows', 'hole_cols', 'hole_inds1d'
        ]

    @staticmethod
    def _get_cache_file_path(det_name, cache_dir):
        """
        Constructs the path to the cache file for a given detector name.

        Parameters:
        det_name (str): Detector name
        cache_dir (str or Path): Directory where cache file is stored

        Returns:
        Path: Full path to the cache file
        """
        return Path(cache_dir) / f"{det_name}_calibc_cache.pkl"

    @staticmethod
    def load(det_name, iface, cache_dir="/dev/shm", logger=None):
        """
        Loads cached CalibConstants attributes into the specified detector interface.

        Parameters:
        det_name (str): Detector name
        iface (object): Detector interface (e.g., det.raw) with a _calibc_ attribute
        cache_dir (str): Directory where pickle files are stored
        logger (Logger): Optional logger

        Returns:
        bool: True if successful, False otherwise
        """
        path = DetectorCacheManager._get_cache_file_path(det_name, cache_dir)
        if not path.exists():
            if logger:
                logger.warning(f"Cache file not found: {path}")
            return False
        try:
            with open(path, 'rb') as f:
                full_cache = pickle.load(f)
            drp_class_name = getattr(iface, "_drp_class_name", None)
            if not drp_class_name or drp_class_name not in full_cache:
                return False

            cached_attrs = full_cache[drp_class_name]

            # If _calibc_ is already available, apply directly
            if hasattr(iface, "_calibc_") and iface._calibc_ is not None:
                for attr, val in cached_attrs.items():
                    setattr(iface._calibc_, attr, val)
                if logger:
                    logger.debug(f"Loaded cache for {det_name}.{drp_class_name} from {path}")
            else:
                # _calibc_ not yet initialized, stash for later use
                iface._calibc_preload_cache = cached_attrs
                if logger:
                    logger.debug(f"Deferred cache load for {det_name}.{drp_class_name} (pending _calibc_ init)")
            return True
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load cache: {e}")
            return False

    def save(self):
        """
        Saves selected attributes of CalibConstants to a pickle file.
        The format is: {drp_class_name: {attr1: val1, attr2: val2, ...}}
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        full_cache = {}
        for drp_class_name in dir(self.det):
            if drp_class_name.startswith("_"):
                continue
            iface = getattr(self.det, drp_class_name, None)
            if self.det_name == "jungfrau" and hasattr(iface, "image"):
                if not hasattr(iface, "_calibc_"):
                    raise RuntimeError(
                        f"DetectorCacheManager: jungfrau {drp_class_name} missing _calibc_; cannot save cache"
                    )
                if iface._calibc_ is None:
                    raise RuntimeError(
                        f"DetectorCacheManager: jungfrau {drp_class_name} _calibc_ is None; cannot save cache"
                    )
            if not hasattr(iface, "_calibc_"):
                continue
            cc = iface._calibc_
            if cc is None:
                continue

            entry = {}
            for attr in self.attrs_to_cache:
                if hasattr(cc, attr):
                    entry[attr] = getattr(cc, attr)
            if entry:
                full_cache[drp_class_name] = entry
            else:
                self.logger.warning(f"Entry for {drp_class_name} is empty")

        t0 = time.monotonic()
        with open(self.cache_file, 'wb') as f:
            pickle.dump(full_cache, f)
        self.logger.info(f"Saved cache to {self.cache_file} in {time.monotonic()-t0:.2f}s.")

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
            self.logger.debug("Computing and caching CalibConstants attributes...")
            try:
                t0 = time.monotonic()
                evt = next(self.det._run.events())
                # trigger image access to populate _calibc_ for all interfaces
                for drp_class_name in dir(self.det):
                    if drp_class_name.startswith("_"):
                        continue
                    iface = getattr(self.det, drp_class_name, None)
                    if hasattr(iface, "image"):
                        try:
                            _ = iface.image(evt)
                        except Exception:
                            pass
                self.logger.debug(f"CalibConstants attributes computed in {time.monotonic()-t0:.2f}s.")
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
