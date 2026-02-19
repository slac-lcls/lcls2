import logging
import time
import json
import os
import numpy as np
import copy

from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import cdict
from psdaq.configdb.det_config import *
from psdaq.cas.xpm_utils import timTxId
import pyrogue as pr
import rogue
import rogue.hardware.axi

from psdaq.utils import enable_epix_uhr_gtreadout_dev
import epix_uhr_gtreadout_dev as epixUhrDev
import surf.protocols.batcher as batcher

class EpixUHRConfigurator:
    DET_SIZE = (4, 168, 192)
    EVENT_BUILDER_TIMEOUT = 0
    DELTA_DELAY = -192
    GAIN_PATH = '/tmp/ePixUHR_GTReadout_default_'
    PLL_PATH = '/tmp/'
    PLL_LABELS = [None, '_temp250', '_2_3_7', '_0_5_7', '_2_3_9', '_0_5_7_v2']
    TIMING_OUT_EN_LIST = [
        'asicR0', 'asicACQ', 'asicSRO', 'asicInj', 'asicGlbRstN',
        'timingRunTrigger', 'timingDaqTrigger', 'acqStart', 'dataSend', '_0', '_1'
    ]
    GAIN_CSV_LIST = [
        '_0_default', '_1_injection_truck', '_2_injection_corners_FHG',
        '_3_injection_corners_AHGLG1', '_4_extra_config', '_5_extra_config',
        '_6_truck2', '_7_on_the_fly'
    ]
    ALG_VERSION = [3, 2, 1]
    SEGLIST = [0, 1]

    def __init__(self):
        self.base = None
        self.chan = None
        self.group = None
        self.orig_cfg = None
        self.seg_ids = None
        self.asics = None
        self.gain_map = np.zeros(self.DET_SIZE)

    @staticmethod
    def _dict_compare(d1, d2, path):
        '''
        Compare two dictionaries and log differences.
        
        Args:
            d1 (dict): First dictionary to compare.
            d2 (dict): Second dictionary to compare.
            path (str): Current path in the dictionary for logging.
        Returns:
            None
        '''
        for k in d1.keys():
            if k in d2.keys():
                if isinstance(d1[k], dict):
                    EpixUHRConfigurator._dict_compare(d1[k], d2[k], path + '.' + k)
                elif (d1[k] != d2[k]):
                    logging.info(f'key[{k}] d1[{d1[k]}] != d2[{d2[k]}]')
            else:
                logging.info(f'key[{k}] not in d1')
        for k in d2.keys():
            if k not in d1.keys():
                logging.info(f'key[{k}] not in d2')

    @staticmethod
    def sanitize_config(src: dict) -> dict:
        '''
        Sanitize configuration dictionary by removing special characters from keys.
        
        Args:
            src (dict): Source configuration dictionary.
        Returns:
            dict: Sanitized configuration dictionary.
        '''
        dst = {}
        for k, v in src.items():
            if isinstance(v, dict):
                v = EpixUHRConfigurator.sanitize_config(v)
            dst[k.replace('[','').replace(']','').replace('(','').replace(')','')] = v
        return dst

    def panel_asic_init(self, det_root: dict, asics: list):
        '''
        Initialize ASICs in the panel.
        Args:
            det_root (dict): Detector root configuration.
            asics (list): List of ASIC identifiers.
        Returns:
            None
        '''
        for asic in asics:
            self.write_to_detector(getattr(det_root.App, f"BatcherEventBuilder{asic}").enable, True)
            self.write_to_detector(getattr(det_root.App, f"BatcherEventBuilder{asic}").Bypass, 0)
            self.write_to_detector(getattr(det_root.App, f"BatcherEventBuilder{asic}").Timeout, 0)
            self.write_to_detector(getattr(det_root.App, f"BatcherEventBuilder{asic}").Blowoff, False)
            self.write_to_detector(getattr(det_root.App, f"FramerAsic{asic}").enable, False)
            self.write_to_detector(getattr(det_root.App, f"FramerAsic{asic}").DisableLane, 0)
            self.write_to_detector(getattr(det_root.App, f"AsicGtData{asic}").enable, True)
            self.write_to_detector(getattr(det_root.App, f"AsicGtData{asic}").gtStableRst, False)

    def panel_init(self, det_root: dict):
        # ... (copy the body of panel_init here, replacing write_to_detector with self.write_to_detector)
        self.write_to_detector(det_root.App.WaveformControl.enable, True)
        # ... (repeat for all other lines in panel_init)

    def epixUHR_init(self, arg, dev='/dev/datadev_0', lanemask=0xf, xpmpv=None, timebase="186M", verbosity=0) -> dict:
        logging.getLogger().setLevel(logging.WARNING)
        logging.info('epixUHR_init')
        self.base = {}
        det_root = epixUhrDev.Root(
            dev=dev, defaultFile=' ', emuMode=False, pollEn=True, initRead=True,
            viewAsic=0, dataViewer=False, numClusters=14, otherViewers=False,
            numOfAsics=4, timingMessage=False, justCtrl=True, loadPllCsv=False,
        )
        det_root.__enter__()
        self.base['cam'] = det_root
        firmwareVersion = det_root.Core.AxiVersion.FpgaVersion.get()
        buildDate = det_root.Core.AxiVersion.BuildDate.get()
        gitHashShort = det_root.Core.AxiVersion.GitHashShort.get()
        logging.info(f'firmwareVersion [{firmwareVersion:x}]')
        logging.info(f'buildDate       [{buildDate}]')
        logging.info(f'gitHashShort    [{gitHashShort}]')
        logging.warning(f'Using timebase {timebase}')
        self.panel_init(det_root)
        # ... (continue with the rest of epixUHR_init, replacing global with self.)
        # ... (return self.base at the end)

    # ... (repeat for all other functions, converting them to methods and replacing global variables with self attributes)

    @staticmethod
    def write_to_detector(var, val):
        if (var.get() != val):
            var.set(val)
            if var.get() != val:
                logging.error(f"Failed to write to detector {var}:{val}")
            else:
                logging.debug(f"File written correctly {var}:{val}")
        else:
            logging.debug(f"Variable already set {var}:{val}")

    # ... (implement other methods as needed)

# Example usage:
# config = EpixUHRConfigurator()
# config.epixUHR_init(...)