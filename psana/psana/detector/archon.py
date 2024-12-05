
#import sys
from psana.detector.detector_impl import DetectorImpl
from amitypes import Array2d
#import logging
#from psana.detector.NDArrUtils import info_ndarr #, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d
from psana.detector.areadetector import AreaDetectorRaw, sgs
import numpy as np


class archon_raw_1_0_0(AreaDetectorRaw):
    def __init__(self, *args, **kwargs): # **kwargs intercepted by AreaDetectorRaw
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
        #print('XXX archon_raw_1_0_0._cmpars: %s' % str(self._cmpars))
        #print('XXX archon_raw_1_0_0._gainfact: %.3f' % self._gainfact)
        #print('XXX calibconst _gain_factor(): %s' % str(self._gain_factor()))
        #print('XXX dir(_seg_geo):', dir(self._seg_geo))
        #self._mask_arr = self._seg_geo.mask_fake()
        #self._ixy = self._seg_geo.get_seg_xy_maps_pix_with_offset()
        #print('XXX dir(_seg_geo):', dir(self._seg_geo))
        #print(info_ndarr(self._ixy[0], 'X pixmap:'))
        #print(info_ndarr(self._ixy[1], 'Y pixmap:'))
        #sys.exit('TEST EXIT')

    def _common_mode(self, frame):
        # courtesy of Phil Hart
        # this is slow because of the python loops, but the
        # camera readout is <1Hz with many rows, so perhaps OK? - cpo
        bankSize = self._totPixelsPerBank
        rows = frame.shape[0]
        for r in range(rows):
            colOffset = 0
            for b in range(self._nbanks):
                # this also corrects the fake pixels themselves
                frame[r, colOffset:colOffset+bankSize] -= \
                    frame[r, colOffset+bankSize-self._fakePixelsPerBank:colOffset+bankSize].mean()
                colOffset += bankSize

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

#    def _mask_fake_v0(self, raw_shape, dtype=np.uint8) -> Array2d:
#        if raw_shape is None: return None
#        rows = raw_shape[0]
#        mbank = np.hstack([np.ones ((rows, self._realPixelsPerBank), dtype=dtype),\
#                           np.zeros((rows, self._fakePixelsPerBank), dtype=dtype),])
#        return np.hstack((self._nbanks * (mbank,)))

    def _mask_fake(self, raw_shape, dtype=np.uint8) -> Array2d:
        """returns mask of shape=(<nrows>,4800), with fake pixels of all banks set to 0"""
        fake1bank = np.zeros((raw_shape[0], self._fakePixelsPerBank), dtype=dtype)
        mask = np.ones(raw_shape, dtype=dtype)
        sr, st = self._realPixelsPerBank, self._totPixelsPerBank # = 264, 300
        for i in range(self._nbanks):
             mask[:,st*i+sr:st*(i+1)] = fake1bank
        return mask

    def _tstamp(self, evt, ibank=0):
        """returns time stamp as a 8-byte integer recorded in the beginning of each fake-pixel bank"""
        raw = self.raw(evt)
        return None if raw is None else self._tstamp_raw(raw, ibank=ibank)

    def _tstamp_raw(self, raw, ibank=0):
        """the same as previous but raw in the input"""
        if raw is None: return None
        sr, st = self._realPixelsPerBank, self._totPixelsPerBank # = 264, 300
        sb = st*ibank+sr
        return np.frombuffer(raw[0,sb:sb+4].tobytes(), dtype=np.uint64)

    def _set_tstamp_pixel_values(self, a, value=0):
        """sets 4 pixel intensities in the beginning of each fake bank to specified value"""
        if a is None: return None
        sr, st = self._realPixelsPerBank, self._totPixelsPerBank # = 264, 300
        for i in range(self._nbanks):
             sb = st*i+sr
             a[0,sb:sb+4] = value
        return a

# EOF
