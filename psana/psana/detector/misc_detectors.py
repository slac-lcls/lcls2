import sys
from psana.detector.detector_impl import DetectorImpl
from amitypes import Array2d
import logging
from psana.detector.NDArrUtils import info_ndarr #, shape_nda_as_3d, reshape_to_3d # shape_as_3d, shape_as_3d
from psana.detector.areadetector import AreaDetectorRaw, sgs
import numpy as np

# create a dictionary that can be used to look up other
# information about an epics variable.  the key in
# the dictionary is the "detname" of the epics variable
# (from "detnames -e") which can be either the true (ugly)
# epics name or a nicer user-defined name. the most obvious
# way to use this information would be to retrieve the
# real epics name from the "nice" user-defined detname.

class epicsinfo_epicsinfo_1_0_0(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
        self._infodict={}
        for c in self._configs:
            if hasattr(c,'epicsinfo'):
                for seg,value in c.epicsinfo.items():
                    names = getattr(value,'epicsinfo')
                    keys = names.keys.split(',')
                    for n in dir(names):
                        if n.startswith('_') or n=='keys': continue
                        if n not in self._infodict: self._infodict[n]={}
                        values = getattr(names,n).split('\n')
                        for k,v in zip(keys,values): self._infodict[n][k]=v
    def __call__(self):
        return self._infodict

class pvdetinfo_pvdetinfo_1_0_0(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
        self._infodict={}
        for c in self._configs:
            if hasattr(c,'pvdetinfo'):
                for seg,value in c.pvdetinfo.items():
                    names = getattr(value,'pvdetinfo')
                    keys = names.keys.split(',')
                    for n in dir(names):
                        if n.startswith('_') or n=='keys': continue
                        if n not in self._infodict: self._infodict[n]={}
                        values = getattr(names,n).split(',')
                        for k,v in zip(keys,values): self._infodict[n][k]=v

    def __call__(self):
        return self._infodict

class pv_raw_1_0_0(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
        self._add_fields()

class encoder_raw_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    def value(self,evt) -> float:
        """
        From Zach:
        “Scale” is copied from the configured “units per encoderValue”
        configuration. So a scale of “6667” means we have 6667 units per count.
        What is the unit? Well, I coded it to be “nanometers per count”
        for the average axis that is controlled in millimeters, since for
        most axes this ends up being an integer, but this axis is controlled
        in microradians, so that “normal case” conversion doesn’t really fit.
        Checking the code, the configured scale for the encoder is 0.0066667
        urads/count. I guess that makes sense since you’d usually multiply
        by 1e-6 to go from mm to nm.

        So to get the “real motor position” you would in general do:
        Position (real units) = encoderValue * scale * 1e-6
        Which works for the current encoderValue and the real position
        shown in the screen.

        Though clearly the scale is rounded in this case, with a true
        value of 2/3 * 1e4

        The Controls group has conventions of mm/urad for linear/rotary
        motion units, so this routine returns urad for the rix mono encoder.
        """
        segments = self._segments(evt)
        if segments is None: return None
        # NOTE: here we only return the first channel of the array to
        # make it easier for the users to use.  If we go to multi-channel in future
        # we could update the version number of the raw data and have
        # another det xface that returns all channels (leaving out the [0]
        # in the "encoderValue" and "scale" below). - cpo 09/28/21
        # note that the order of operations matters here: if
        # we multiply the two numpy array values together we can overflow
        # a uint32.  so convert to float first.
        return (segments[0].encoderValue[0]*1e-6)*segments[0].scale[0]

class encoder_raw_2_0_0(encoder_raw_0_0_1):
    def __init__(self, *args):
        super().__init__(*args)
    def value(self,evt) -> float:
        """
        The Controls group has conventions of mm/urad for linear/rotary
        motion units, so this routine returns urad for the rix mono encoder.
        """
        segments = self._segments(evt)
        if segments is None: return None
        # if scaleDenom > 0, multiply by (float)scale/(float)scaleDenom.
        # Otherwise, inherit from the parent class.
        if (segments[0].scaleDenom[0] > 0):
            return segments[0].encoderValue[0]*(float(segments[0].scale[0])/float(segments[0].scaleDenom[0]))
        else:
            return super().value(evt)

class encoder_raw_2_1_0(encoder_raw_2_0_0):
    def __init__(self, *args):
        super().__init__(*args)
    def value(self,evt) -> float:
        """
        Version 2.1.0 adds innerCount field.  The return value is unaffected.
        """
        return super().value(evt)

class encoder_raw_3_0_0(DetectorImpl):
   def __init__(self, *args):
       super().__init__(*args)
   def value(self,evt) -> float:
       """
       Version 3.0.0 addresses fields as scalars, not as arrays.
       """
       segments = self._segments(evt)
       if segments is None: return None
       # if scaleDenom > 0, multiply by (float)scale/(float)scaleDenom.
       # Otherwise, multiply by 1.0.
       if (segments[0].scaleDenom > 0):
           return segments[0].encoderValue*(float(segments[0].scale)/float(segments[0].scaleDenom))
       else:
           return segments[0].encoderValue*1.0

class hrencoder_raw_0_1_0(DetectorImpl):
    """High rate encoder.

    The hrencoder detector returns 4 fields:
        position - The encoder value/position.
        missedTrig_cnt - Missed triggers
        error_cnt - Number of errors.
        latches - Only 3 bits of this integer correspond to the latch status.
    """
    def __init__(self, *args):
        super().__init__(*args)

    def value(self, evt) -> float:
        segments = self._segments(evt)

        if segments is None:
            return None

        return segments[0].position

class encoder_interpolated_3_0_0(encoder_raw_3_0_0):
    def __init__(self, *args):
        super().__init__(*args)
    def value(self,evt) -> float:
        """
        Version 3.0.0 addresses fields as scalars, not as arrays.
        """
        return super().value(evt)

class archon_raw_1_0_0(AreaDetectorRaw):
    def __init__(self, *args, **kwargs): # **kwargs intercepted by AreaDetectorRaw
        super().__init__(*args, **kwargs)
        self._nbanks = 16
        self._fakePixelsPerBank = 36
        self._realPixelsPerBank = 264
        self._totPixelsPerBank = self._fakePixelsPerBank+self._realPixelsPerBank
        self._totRealPixels = self._realPixelsPerBank*self._nbanks
        self._gainfact = self._kwargs.get('gainfact', 3.65) # 3.65eV/ADU
        self._cmpars = self._kwargs.get('cmpars', None)
        self._seg_geo = sgs.Create(segname='ARCHON:V1', detector=self)
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-archon.data'
        self._geo = self._det_geo()
        #print('XXX archon_raw_1_0_0._cmpars: %s' % str(self._cmpars))
        #print('XXX archon_raw_1_0_0._gainfact: %.3f' % self._gainfact)
        #print('XXX calibconst _gain_factor(): %s' % str(self._gain_factor()))

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
        #print('XXX: archon_raw_1_0_0.image: self._geo:', self._geo)

        return self._arr_to_image(c) #### <<<<<<<<<=======================

        # Use local conversion of raw-like array to image or SegmentGeometry object
        if self._geo is None:
            return self._arr_to_image(c)
        # Use GeometryAccess object
        X, Y, Z = self._geo.get_pixel_coords(oname=None, oindex=0, do_tilt=True, cframe=0)
        print(info_ndarr(X, 'X coords:'))
        print(info_ndarr(Y, 'Y coords:'))
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

# Test
class justafloat_simplefloat32_1_2_4(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
    def value(self,evt) -> float:
        return self._segments(evt)[0].valfloat32

# EOF
