
"""
import psana.detector.utils_ami as ua

    def __init__(self):
        self.counter = 0
        self.cc = None

    def on_event(self, raw, config, calibconst, *args, **kwargs):
        self.counter += 1
        if self.cc is None:
            self.cc = ua.calib_components(calibconst, config)
        cc = self.cc

        ctypes   = cc.calib_types()
        npanels  = cc.number_of_panels()
        peds     = cc.pedestals()    # OR cc.calib_constants('pedestals')
        gain     = cc.gain()         # ADU/keV
        gfactor  = cc.gain_factor()  # keV/ADU
        status   = cc.status()
        comode   = cc.common_mode()
        trbit_p0 = cc.trbit_for_panel(0)
        ascfg_p0 = cc.asicPixelConfig_for_panel(0)
        mask_st  = cc.mask_from_status(status_bits=0xffff, gain_range_inds=None)
        mask     = cc.mask(status=True)
        dettype  = cc.dettype()
        cbitscfg = cc.cbits_config_detector()
        cbitstot = cc.cbits_config_and_data_detector(raw, cbitscfg)
        gmap     = cc.gain_maps_epix10ka_any(raw)
        peds_ev  = cc.event_pedestals(raw)
        gfac_ev  = cc.event_gain_factor(raw)
        calib    = cc.calib(raw, cmpars=(0, 7, 300, 10))  # **kwa - for non-default mask parameters
        cmcorr   = cc.common_mode_correction(raw, cmpars=(0, 7, 300, 10))  # **kwa

        print('== Event %04d ==' % self.counter)
        print('calib_types', ctypes)
        # print('calib_metadata', cc.calib_metadata('pedestals'))
        print(ua.info_ndarr(peds, 'pedestals'))
        print(ua.info_ndarr(cc.gain(), 'gain'))
        print(ua.info_ndarr(gfactor, 'gain_factor'))
        print(ua.info_ndarr(status, 'status'))
        print('common_mode from caliconst', str(comode))
        print('number_of_panels', npanels)
        print('trbit_for_panel(0)', trbit_p0)
        print(ua.info_ndarr(ascfg_p0, 'asicPixelConfig_for_panel(0)'))
        print(ua.info_ndarr(raw, 'raw'))
        print(ua.info_ndarr(mask_st, 'mask from status'))
        print(ua.info_ndarr(mask, 'mask'))
        print('dettype', dettype)
        print(ua.info_ndarr(cbitscfg, 'cbitscfg'))
        print(ua.info_ndarr(cbitstot, 'cbitstot'))
        print(ua.info_ndarr(gmap, 'gmap'))
        print(ua.info_ndarr(peds_ev, 'peds_ev'))
        print(ua.info_ndarr(gfac_ev, 'gfac_ev'))
        print(ua.info_ndarr(calib, 'calib'))
        print(ua.info_ndarr(cmcorr, 'cmcorr'))
        # img = cmcorr[0, 144:, :192]
        img = ua.psu.table_nxn_epix10ka_from_ndarr(cmcorr)
        print(ua.info_ndarr(img, 'img'))
        return img

# for epixquad
ami-local -b 1 -f interval=1 psana://exp=ueddaq02,run=569,dir=/cds/data/psdm/prj/public01/xtc

# for epixhr:
ami-local -b 1 -f interval=1 psana://exp=rixx45619,run=121,dir=/cds/data/psdm/prj/public01/xtc
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
from psana.detector.NDArrUtils import divide_protected, info_ndarr
import psana.detector.UtilsEpix10ka as ue
from psana.detector.mask_algos import MaskAlgos, DTYPE_MASK, DTYPE_STATUS
import psana.pyalgos.generic.PSUtils as psu

gain_maps_epix10ka_any, event_constants_for_gmaps, cbits_config_epix10ka, cbits_config_epixhr2x2 =\
ue.gain_maps_epix10ka_any, ue.event_constants_for_gmaps, ue.cbits_config_epix10ka, ue.cbits_config_epixhr2x2


def is_none(val, msg):
    s = val is None
    if s: logger.debug(msg)
    return s


class calib_components():

    def __init__(self, calibconst, config):
        self.calibconst = calibconst
        self.config = config   # config={0: <psana.container.Container...>, 1:...}
        self._gfactor = None
        self.omask = None
        self._dettype = None
        self.cbits_cfg = None
        self._store_ = None
        self._data_bit_mask = None

    def calib_types(self):
        """returns list of available calib types, e.g.
        ['pixel_rms', 'geometry', 'pedestals', 'pixel_status', 'pixel_gain']"""
        if isinstance(self.calibconst, dict): return self.calibconst.keys()
        else:
            logger.debug('calibconst IS NOT DICT: %s' % (str(self.calibconst)))
            return None

    def _calibconst_tuple_for_ctype(self, ctype='pedestals'):
        """returns list of calib constants array and metadata"""
        tupcc = self.calibconst.get(ctype, None)
        if isinstance(tupcc, tuple): return tupcc
        else:
            logger.debug('calibconst[%s] = %s IS NOT TUPLE' % (ctype, str(tupcc)))
            return None

    def calib_constants(self, ctype='pedestals'):
        tcc = self._calibconst_tuple_for_ctype(ctype)
        return tcc[0] if tcc is not None else None

    def calib_metadata(self, ctype='pedestals'):
        tcc = self._calibconst_tuple_for_ctype(ctype)
        return tcc[1] if tcc is not None else None

    def common_mode(self):
        """returns list of common mode parameters"""
        return self.calib_constants('common_mode')

    def pedestals(self):
        """pedestals array shape=(7, <number-of-panels>, 352, 384)"""
        return self.calib_constants('pedestals')

    def gain(self):
        """ADU/keV, array shape=(7, <number-of-panels>, 352, 384)"""
        return self.calib_constants('pixel_gain')

    def status(self):
        """array shape=(7, <number-of-panels>, 352, 384)"""
        return self.calib_constants('pixel_status')

    def gain_factor(self):
        """keV/ADU, array shape=(7, <number-of-panels>, 352, 384)"""
        if self._gfactor is None:
            g = self.gain()
            self._gfactor = divide_protected(np.ones_like(g), g)
        return self._gfactor

    def number_of_panels(self):
        return len(self.config)

    def trbit_for_panel(self, i):
        """returns list of trbit for 4 ASICs per panel, e.g. [1,1,1,1]"""
        return self.config[i].config.trbit

    def asicPixelConfig_for_panel(self, i):
        """returns list of asicPixelConfig per panel, array shape:(4, 176, 192) for 4 ASICs per panel"""
        return self.config[i].config.asicPixelConfig

    def mask_from_status(self, status_bits=0xffff, gain_range_inds=None, dtype=DTYPE_MASK, **kwa):
        if self.omask is None: self.omask = MaskAlgos(self.calibconst, **kwa)
        return self.omask.mask_from_status(status_bits, gain_range_inds, dtype, **kwa)

    def mask(self, status=True, neighbors=False, edges=False, center=False, calib=False, umask=None, force_update=False, dtype=DTYPE_MASK, **kwa):
        if self.omask is None: self.omask = MaskAlgos(self.calibconst, **kwa)
        return self.omask.mask(status=status, neighbors=neighbors, edges=edges, center=center, calib=calib, umask=umask, dtype=dtype, **kwa)

    def dettype(self):
        """returns cached (str) for detector type dettype, e.g. 'epix10ka' or 'epixhr'"""
        if self._dettype is None:
            metad = self.calib_metadata(ctype='pedestals')
            if metad is None: return None
            self._dettype = metad.get('dettype', None)
            if 'epixhr' in self._dettype: self._dettype = 'epixhr'  # trancates epixhr2x2
        return self._dettype

    def data_bit_mask(self):
        """returns cached _data_bit_mask"""
        if self._data_bit_mask is None:
            dettype = self.dettype()
            assert dettype in ('epix10ka', 'epixhr'), 'implemented for listed detect types only'
            self._data_bit_mask = {'epix10ka':ue.M14, 'epixhr':ue.M15}.get(dettype, None)
        return self._data_bit_mask

    def cbits_config_segment(self, cob):
        """analog of epix_base._cbits_config_segment, where cob is self.config[<segment-number>].config """
        dettype = self.dettype()
        return cbits_config_epixhr2x2(cob, shape=(288, 384)) if dettype == 'epixhr' else\
               cbits_config_epix10ka(cob, shape=(352, 384)) if dettype == 'epix10ka' else\
               None

    def cbits_config_detector_alg(self):
        """analog of epix_base._cbits_config_detector"""
        if self.config is None: return None
        lst_cbits = [self.cbits_config_segment(v.config) for k,v in self.config.items()]
        return np.stack(tuple(lst_cbits))

    def cbits_config_detector(self):
        if self.cbits_cfg is None:
           self.cbits_cfg = self.cbits_config_detector_alg()
        return self.cbits_cfg

    def cbits_config_and_data_detector(self, raw, cbits):
        """analog of UtilsEpix10ka method cbits_config_and_data_detector"""
        dettype = self.dettype()
        assert dettype in ('epix10ka', 'epixhr'), 'implemented for listed detector types only, unknown type %s' % str(dettype)
        data_gain_bit = {'epix10ka':ue.B14, 'epixhr':ue.B15}.get(dettype, None)
        gain_bit_shift = {'epix10ka':9, 'epixhr':10}.get(dettype, None)
        return ue.cbits_config_and_data_detector_alg(raw, cbits, data_gain_bit, gain_bit_shift)

    def gain_maps_epix10ka_any(self, raw):
        """analog of UtilsEpix10ka method gain_maps_epix10ka_any"""
        cbitscfg = self.cbits_config_detector()
        cbitstot = self.cbits_config_and_data_detector(raw, cbitscfg)
        return ue.gain_maps_epix10ka_any_alg(cbitstot)

    def event_constants_for_gmaps(gmaps, cons, default=0):
        return ue.event_constants_for_gmaps(gmaps, cons, default)

    def event_pedestals(self, raw):
        """ returns per-event  pedestals, shape=(<number-of-panels>, <2-d-panel-shape>)"""
        return ue.event_constants_for_gmaps(self.gain_maps_epix10ka_any(raw), self.pedestals(), default=0)

    def event_gain_factor(self, raw):
        """returns per-event gain_factor, shape=(<number-of-panels>, <2-d-panel-shape>)"""
        return ue.event_constants_for_gmaps(self.gain_maps_epix10ka_any(raw), self.gain_factor(), default=1)

    def event_gain(self, raw):
        """returns per-event gain, shape=(<number-of-panels>, <2-d-panel-shape>)"""
        return ue.event_constants_for_gmaps(self.gain_maps_epix10ka_any(raw), self.gain(), default=1)


    def calib(self, raw, cmpars=None, **kwa):
        """equivalent of the UtilsEpix10ka.py method calib_epix10ka_any"""

        if is_none(raw, 'in calib raw is None - return None'): return None

        gmaps = self.gain_maps_epix10ka_any(raw)
        if is_none(gmaps, 'in calib gmaps is None - return None'): return None

        store = Storage(self, cmpars, **kwa) if self._store_ is None else self._store_
        mask = store.mask
        factor = ue.event_constants_for_gmaps(gmaps, store.gfac, default=1)  # 3d gain factors
        pedest = ue.event_constants_for_gmaps(gmaps, store.peds, default=0)  # 3d pedestals

        store.counter += 1
        if not store.counter%100: ue.print_gmaps_info(gmaps)

        if is_none(pedest, 'in calib pedest is None - return None'): return None
        if is_none(factor, 'in calib factor is None - return None'): return None

        arrf = np.array(raw & self.data_bit_mask(), dtype=np.float32) - pedest

        if store.cmpars is not None:
            ue.common_mode_epix_multigain_apply(arrf, gmaps, store)

        is_none(mask, 'in calib mask is None - not applied')

        return arrf * factor if mask is None else arrf * factor * mask


    def common_mode_correction(self, raw, cmpars=None, **kwa):
        """ Returns common mode correction for the raw-peds.
        """
        if is_none(raw, 'in calib raw is None - return None'): return None

        gmaps = self.gain_maps_epix10ka_any(raw)
        if is_none(gmaps, 'in calib gmaps is None - return None'): return None

        store = Storage(self, cmpars, **kwa) if self._store_ is None else self._store_

        pedest = ue.event_constants_for_gmaps(gmaps, store.peds, default=0)  # 3d pedestals

        arrf = np.array(raw & self.data_bit_mask(), dtype=np.float32) - pedest
        arrf0 = arrf.copy()

        if store.cmpars is not None:
            ue.common_mode_epix_multigain_apply(arrf, gmaps, store)

        return (arrf - arrf0) * store.mask


class Storage:
    def __init__(self, calibcomps, cmpars=None, **kwa):
        """analogy of UtilsEpix10ka.py class Storage
        Holds cached parameters for common mode correction of the epix multi-gain getectors.

        Parameters
        ----------
        - counter (int) - event counter
        - gain (ndarray (7, <nsegs>, 352, 384)) - gains from calibration constants
        - peds (ndarray (7, <nsegs>, 352, 384)) - pedestals from calibration constants
        - shape_as_daq (tuple) - shape (<nsegs>, 352, 384) from calibration constants
        - mask - (ndarray (<nsegs>, 352, 384)) - mask retreived from calibration constants
        - arr1 - (ndarray (<nsegs>, 352, 384)) - ones with shape_as_daq
        - cmpars (int or tuple) - user defined or from calibration constants if None, 0 - cm correction is turrned off
        """

        logger.info('create store with cached parameters')

        calibcomps._store_ = self  # self preservation
        self.counter = -1

        self.gain = calibcomps.gain()      # - 4d gains  (7, <nsegs>, 352, 384)
        self.peds = calibcomps.pedestals() # - 4d pedestals
        self.shape_as_daq = self.peds.shape[1:]
        self.gfac = divide_protected(np.ones_like(self.gain), self.gain)
        self.arr1 = np.ones(self.shape_as_daq, dtype=np.int8)

        self.mask = calibcomps.mask(**kwa)
        if self.mask is None: self.mask = calibcomps._mask_from_status(**kwa)
        if self.mask is None: self.mask = np.ones(self.shape_as_daq, dtype=DTYPE_MASK)

        self.cmpars = calibcomps.common_mode() if cmpars is None else cmpars

        logger.info('\n  shape_as_daq %s' % str(self.shape_as_daq)\
                    +info_ndarr(self.gain, '\n  gain')\
                    +info_ndarr(self.peds, '\n  peds')\
                    +info_ndarr(self.gfac, '\n  gfac')\
                    +info_ndarr(self.mask, '\n  mask')\
                    +'\n  common-mode correction parameters cmpars: %s' % str(self.cmpars))

# EOF
