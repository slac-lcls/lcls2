
#import sys
import numpy as np
from time import time
import logging
logger = logging.getLogger(__name__)
from psana.detector.detector_impl import DetectorImpl
from amitypes import Array2d
from psana.detector.areadetector import AreaDetectorRaw, sgs
#from psana.detector.NDArrUtils import info_ndarr #, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d


class archon_raw_1_0_0(AreaDetectorRaw):
    def __init__(self, *args, **kwargs): # **kwargs intercepted by AreaDetectorRaw
        """1_0_0: daq raw array consists of 16 banks, bank has variable number off rows.
           bank has 264 columns of real, and 36 columns af fake pixels, described by ARCHON:V1 geometry
        """
        super().__init__(*args, **kwargs)
        self._nbanks = 16
        self._fakePixelsPerBank = 36
        self._realPixelsPerBank = 264
        self._totPixelsPerBank = self._fakePixelsPerBank+self._realPixelsPerBank
        self._totRealPixels = self._realPixelsPerBank*self._nbanks
        self._gainfact = self._kwargs.get('gainfact', 3.65) # 3.65eV/ADU
        self._cmpars = self._kwargs.get('cmpars', (1,0,0))
        self._seg_geo = sgs.Create(segname='ARCHON:V1', detector=self)
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-archon.data'
        self._geo = self._det_geo()

    def _common_mode_v0(self, frame):
        # courtesy of Phil Hartz
        # this is slow because of the python loops, but the
        # camera readout is <1Hz with many rows, so perhaps OK? - cpo
        logger.debug('_common_mode_v0')
        bankSize = self._totPixelsPerBank
        rows = frame.shape[0]
        for r in range(rows):
            colOffset = 0
            for b in range(self._nbanks):
                # this also corrects the fake pixels themselves
                frame[r, colOffset:colOffset+bankSize] -= \
                    frame[r, colOffset+bankSize-self._fakePixelsPerBank:colOffset+bankSize].mean()
                colOffset += bankSize

    def _common_mode_v1(self, frame):
        """the same as above with short indices and test for median"""
        logger.debug('_common_mode_v1')
        sf, st = self._fakePixelsPerBank, self._totPixelsPerBank # = 264, 36, 300
        rows = frame.shape[0]
        for r in range(rows):
            for b in range(self._nbanks):
                c0 = st*b
                #med = np.median(frame[r,c0+st-sf:c0+st])  # returns scalar, axis=1)
                mean = frame[r, c0+st-sf:c0+st].mean()
                frame[r, c0:c0+st] -= mean

    def _common_mode(self, frame):
        """median for entire array of fake pixels, axis=1"""
        logger.debug('_common_mode')
        t0_sec = time()
        sr, sf, st = self._realPixelsPerBank, self._fakePixelsPerBank, self._totPixelsPerBank # = 264, 36, 300
        rows = frame.shape[0]
        for i in range(self._nbanks):
            c0 = st*i
            med = np.median(frame[:,c0+st-sf:c0+st], axis=1)
            #med = np.mean(frame[:,c0+st-sf:c0+st], axis=1)
            st_of_med = np.array(st*(tuple(med),)).T # this also corrects the fake pixels themselves
            frame[:,c0:c0+st] -= st_of_med
        logger.debug('_common_mode time: %.3f sec st_of_med.shape: %s' % (time()-t0_sec, str(st_of_med.shape)))

    def raw(self, evt) -> Array2d:
        segs = self._segments(evt)
        if segs is None: return None
        return segs[0].value

    def calib(self, evt) -> Array2d:
        raw = self.raw(evt)
        if raw is None: return None
        peds = self._calibconst['pedestals'][0]
        if peds is None:
            logging.warning('no archon pedestals')
            return raw
        if peds.shape != raw.shape:
            logging.warning(f'incorrect archon pedestal shape: {peds.shape}, raw data shape: {raw.shape}')
            return raw
        cal = raw-peds
        if self._cmpars is not None:
            self._common_mode(cal)
            #self._common_mode_v1(cal)
        if self._gainfact != 1:
            cal *= self._gainfact
        return cal

    def image(self, evt, nda=None, **kwa) -> Array2d:
        # Make a copy of the data.  would be faster to np.slice, but
        # mpi4py's efficient Reduce/Gather methods don't work
        # with non-contiguous arrays.  but mpi4py does give an error
        # when this happens so could change this. - cpo
        c = self.calib(evt) if nda is None else nda
        if c is None:
            return None
        return self._arr_to_image(c) #### <== ~70ms

    def _arr_to_image_v0(self, c) -> Array2d:
        """converts input array of shape=(<nrows>,4800) to image array of shape=(<nrows>,4224), discarding fake pixels"""
        image = np.empty_like(c, shape=(c.shape[0],self._totRealPixels))
        for i in range(self._nbanks):
            size = self._realPixelsPerBank
            startNoFakePix = i*size
            startSkipFakePix = i*self._totPixelsPerBank
            image[:,startNoFakePix:startNoFakePix+size] \
                = c[:,startSkipFakePix:startSkipFakePix+size]
        return image

    def _arr_to_image(self, a) -> Array2d:
        """converts input array of shape=(<nrows>,4800) to image array of shape=(<nrows>,4224), discarding fake pixels"""
        if a is None: return None
        sr, st = self._realPixelsPerBank, self._totPixelsPerBank # = 264, 300
        return np.hstack([a[:,st*i:st*i+sr] for i in range(self._nbanks)])

    def _mask_fake(self, raw_shape, dtype=np.uint8) -> Array2d:
        """returns mask of shape=(<nrows>,4800), with fake pixels of all banks set to 0"""
        fake1bank = np.zeros((raw_shape[0], self._fakePixelsPerBank), dtype=dtype)
        mask = np.ones(raw_shape, dtype=dtype)
        sr, st = self._realPixelsPerBank, self._totPixelsPerBank # = 264, 300
        for i in range(self._nbanks):
             mask[:,st*i+sr:st*(i+1)] = fake1bank
        return mask

    def _tstamp(self, evt):
        """returns time stamp as a 8-byte integer recorded in the beginning of each fake-pixel bank"""
        raw = self.raw(evt)
        return None if raw is None else self._tstamp_raw(raw)

    def _tstamp_raw(self, raw):
        """the same as previous but raw passed in the input"""
        return None if raw is None else\
               np.frombuffer(raw[0,0:4].tobytes(), dtype=np.uint64)

    def _set_tstamp_pixel_values(self, a, value=0):
        """sets 4 pixel intensities in the beginning of each fake bank to specified value"""
        if a is None: return None
        a[0,0:4] = value
        return a


class archon_raw_1_0_1(AreaDetectorRaw):
    def __init__(self, *args, **kwargs): # **kwargs intercepted by AreaDetectorRaw
        """In 1_0_1 bank fake pixel columns ahead of real:
           daq raw array consists of 16 banks, bank has variable number off rows.
           bank has 36 columns af fake, and 264 columns of real pixels, described by ARCHON:V2 geometry
        """
        super().__init__(*args, **kwargs)
        self._nbanks = 16
        self._fakePixelsPerBank = 36
        self._realPixelsPerBank = 264
        self._totPixelsPerBank = self._fakePixelsPerBank+self._realPixelsPerBank
        self._totRealPixels = self._realPixelsPerBank*self._nbanks
        self._gainfact = self._kwargs.get('gainfact', 3.65) # 3.65eV/ADU
        self._cmpars = self._kwargs.get('cmpars', (1,0,0))
        self._geo = None

    def _init_geometry(self, shape):
        """delayed geometry initialization when raw.shape is available in det.raw.raw"""
        logger.info('_init_geometry for raw.shape %s' % str(shape))
        self._seg_geo = sgs.Create(segname='ARCHON:V2', detector=self, shape=shape)
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-archon.data'
        self._geo = self._det_geo()

    def _common_mode_v0(self, frame):
        """cm from fake with loop over rows"""
        logger.debug('_common_mode_v0')
        sf, st = self._fakePixelsPerBank, self._totPixelsPerBank # = 264, 36, 300
        rows = frame.shape[0]
        for r in range(rows):
            for i in range(self._nbanks):
                # this also corrects the fake pixels themselves
                c0 = st*i
                #cmcorr = np.median(frame[r,c0+st-sf:c0+st])  # returns scalar # NO axis=1
                cmcorr = frame[r, c0:c0+sf].mean() # returns scalar
                frame[r, c0:c0+st] -= cmcorr

    def _common_mode(self, frame):
        """cm as median for entire fake, axis=1"""
        logger.debug('_common_mode')
        t0_sec = time()
        sr, sf, st = self._realPixelsPerBank, self._fakePixelsPerBank, self._totPixelsPerBank # = 264, 36, 300
        rows = frame.shape[0]
        for i in range(self._nbanks):
            c0 = st*i
            med = np.median(frame[:,c0:c0+sf], axis=1)
            st_of_med = np.array(st*(tuple(med),)).T
            frame[:,c0:c0+st] -= st_of_med
            #st_of_med = np.array((st-sf)*(tuple(med),)).T
            #frame[:,c0+sf:c0+st] -= st_of_med
        logger.debug('_common_mode time: %.3f sec st_of_med.shape: %s' % (time()-t0_sec, str(st_of_med.shape)))

    def raw(self, evt) -> Array2d:
        segs = self._segments(evt)
        if segs is None: return None
        r = segs[0].value
        if self._geo is None and r is not None:
            self._init_geometry(r.shape)
        return r

    def calib(self, evt) -> Array2d:
        raw = self.raw(evt)
        if raw is None: return None
        peds = self._calibconst['pedestals'][0]
        if peds is None:
            logging.warning('no archon pedestals')
            return raw
        if peds.shape != raw.shape:
            logging.warning(f'incorrect archon pedestal shape: {peds.shape}, raw data shape: {raw.shape}')
            return raw
        cal = raw-peds
        #cal = np.array(raw, dtype=float) #peds
        if self._cmpars is not None:
            self._common_mode(cal)
            #self._common_mode_v0(cal)
        if self._gainfact != 1:
            cal *= self._gainfact
        return cal

    def image(self, evt, nda=None, **kwa) -> Array2d:
        # Make a copy of the data.  would be faster to np.slice, but
        # mpi4py's efficient Reduce/Gather methods don't work
        # with non-contiguous arrays.  but mpi4py does give an error
        # when this happens so could change this. - cpo
        c = self.calib(evt) if nda is None else nda
        if c is None:
            return None

        return self._arr_to_image(c) #### <<<<<<<<<======================= ~70ms

        # Use local conversion of raw-like array to image or SegmentGeometry object
        #from psana.pyalgos.generic.Graphics import img_from_pixel_arrays
        #return img_from_pixel_arrays(self._ixy[1], self._ixy[0], weights=c, dtype=np.float32, mask_arr=self._mask_arr) # ~300ms

        #if self._geo is None:
        #    return self._arr_to_image(c)
        ## Use GeometryAccess object
        #X, Y, Z = self._geo.get_pixel_coords(oname=None, oindex=0, do_tilt=True, cframe=0)
        #print(info_ndarr(X, 'X coords:'))
        #print(info_ndarr(Y, 'Y coords:'))
        #sys.exit('TEST EXIT')


    def _arr_to_image(self, a) -> Array2d:
        """converts input array of shape=(<nrows>,4800) to image array of shape=(<nrows>,4224), discarding fake pixels"""
        if a is None: return None
        sf, st = self._fakePixelsPerBank, self._totPixelsPerBank # = 36, 300
        return np.hstack([a[:,st*i+sf:st*(i+1)] for i in range(self._nbanks)])

    def _mask_fake(self, raw_shape, dtype=np.uint8) -> Array2d:
        """returns mask of shape=(<nrows>,4800), with fake pixels of all banks set to 0"""
        sf, st = self._fakePixelsPerBank, self._totPixelsPerBank # = 36, 300
        fake1bank = np.zeros((raw_shape[0], sf), dtype=dtype)
        mask = np.ones(raw_shape, dtype=dtype)
        for i in range(self._nbanks):
             mask[:,st*i:st*i+sf] = fake1bank
        return mask

    def _tstamp(self, evt):
        """returns time stamp as a 8-byte integer recorded in the beginning of each fake-pixel bank"""
        raw = self.raw(evt)
        return None if raw is None else self._tstamp_raw(raw)

    def _tstamp_raw(self, raw):
        """the same as previous but raw passed in the input"""
        return None if raw is None else\
               np.frombuffer(raw[0,0:4].tobytes(), dtype=np.uint64)[0]

    def _set_tstamp_pixel_values(self, a, value=0):
        """sets 4 pixel intensities in the beginning of each fake bank to specified value"""
        if a is None: return None
        a[0,0:4] = value
        return a

# EOF
