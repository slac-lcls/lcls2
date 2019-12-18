#----------

"""
Class :py:class:`WFPeaks` a set of methods to find peaks in waveforms
========================================================================

Usage ::

    # Import and create object
    #--------------------------
    from psana.hexanode.WFPeaks import WFPeaks

    # **kwargs - contains peak-finder parameters, see test_WFPeaks() below.
    peaks = WFPeaks(**kwargs)

    # Access methods
    #----------------
    # wfs,wts - input arrays of waveforms and sample times, shape=(nchannels, nsamples)
    peaks.proc_waveforms(wfs, wts)

    # get all-in-one:
    nhits, pkinds, pkvals, pktsec = peaks(wfs,wts)

    # or get individually:
    nhits = peaks.number_of_hits(wfs, wts)
    pktsec = peaks.peak_times_ns(wfs, wts)
    pkvals = peaks.peak_values(wfs, wts)
    pkinds = peaks.peak_indexes(wfs, wts)

Created on 2019-11-07 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np
import psana.pyalgos.generic.Utils as gu
from psana.pyalgos.generic.NDArrUtils import print_ndarr
from ndarray import wfpkfinder_cfd

#----------

class WFPeaks :

    def __init__(self, **kwargs) :
        """Waveform peak finder wrapper.
           - wf digitizer channels (0,1,2,3,4) should be ordered for u1,u2,v1,v2[,w1,w2],mcp, respectively
        """
        logger.debug(gu.str_kwargs(kwargs, title='WFPeaks input parameters:'))
        self.set_wf_peak_finder_parameters(**kwargs)

        self._wfs_old = None

#----------

    def set_wf_peak_finder_parameters(self, **kwargs) :

        self.BASE        = kwargs.get('cfd_base',          0.)
        self.THR         = kwargs.get('cfd_thr',        -0.05)
        self.CFR         = kwargs.get('cfd_cfr',         0.85)
        self.DEADTIME    = kwargs.get('cfd_deadtime',    10.0)
        self.LEADINGEDGE = kwargs.get('cfd_leadingedge', True)
        self.IOFFSETBEG  = kwargs.get('cfd_ioffsetbeg',  1000)
        self.IOFFSETEND  = kwargs.get('cfd_ioffsetend',  2000)
        self.WFBINBEG    = kwargs.get('cfd_wfbinbeg',    6000)
        self.WFBINEND    = kwargs.get('cfd_wfbinend',   30000)

        self.NUM_CHANNELS= kwargs.get('numchs',5)
        self.NUM_HITS    = kwargs.get('numhits',16)

#----------

    def _init_arrays(self) :
        self._number_of_hits = np.zeros((self.NUM_CHANNELS), dtype=np.int)
        self._pkvals = np.zeros((self.NUM_CHANNELS,self.NUM_HITS), dtype=np.double)
        self._pkinds = np.zeros((self.NUM_CHANNELS,self.NUM_HITS), dtype=np.uint32)
        self._pktsec = np.zeros((self.NUM_CHANNELS,self.NUM_HITS), dtype=np.double)

#----------

    def proc_waveforms(self, wfs, wts) :
        """
        """
        # if waveforms are already processed
        if wfs is self._wfs_old : return

        self._init_arrays()
 
        #print_ndarr(wfs, '  waveforms : ', last=4)
        assert (self.NUM_CHANNELS==wfs.shape[0]),\
               'expected number of channels in not consistent with waveforms array shape'

        offsets = wfs[:,self.IOFFSETBEG:self.IOFFSETEND].mean(axis=1)
        #print('  XXX offsets: %s' % str(offsets))

        self.wfsprep = wfs[:,self.WFBINBEG:self.WFBINEND] - offsets.reshape(-1, 1) # subtract wf-offset
        self.wtsprep = wts[:,self.WFBINBEG:self.WFBINEND] # sec

        for ch in range(self.NUM_CHANNELS) :

            wfch = self.wfsprep[ch,:]
            wtch = self.wtsprep[ch,:]

            npeaks = wfpkfinder_cfd(wfch, self.BASE, self.THR, self.CFR, self.DEADTIME, self.LEADINGEDGE,\
                                    self._pkvals[ch,:], self._pkinds[ch,:])
            #print(' npeaks:', npeaks)
            #assert (npeaks<self.NUM_HITS), 'number of found peaks exceeds reserved array shape'
            if npeaks>=self.NUM_HITS : npeaks = self.NUM_HITS
            self._number_of_hits[ch] = npeaks
            self._pktsec[ch, :npeaks] = wtch[self._pkinds[ch, :npeaks]] #sec

        self._wfs_old = wfs

#----------

    def waveforms_preprocessed(self, wfs, wts) :
        """Returns preprocessed waveforms for selected range [WFBINBEG:WFBINEND];
           wfsprep[NUM_CHANNELS,WFBINBEG:WFBINEND] - intensities with subtracted mean evaluated
           wtsprep[NUM_CHANNELS,WFBINBEG:WFBINEND] - times in [sec] like raw data 
        """
        self.proc_waveforms(wfs, wts)
        return self.wfsprep, self.wtsprep

#----------

    def number_of_hits(self, wfs, wts) :
        self.proc_waveforms(wfs, wts)
        return self._number_of_hits

    def peak_times_sec(self, wfs, wts) :
        self.proc_waveforms(wfs, wts)
        return self._pktsec

    def peak_indexes(self, wfs, wts) :
        self.proc_waveforms(wfs, wts)
        return self._pkinds

    def peak_values(self, wfs, wts) :
        self.proc_waveforms(wfs, wts)
        return self._pkvals

    def __call__(self, wfs, wts) :
        self.proc_waveforms(wfs, wts)
        return self._number_of_hits,\
               self._pkinds,\
               self._pkvals,\
               self._pktsec
#----------

    def __del__(self) :
        pass

#----------
#----------
#----------

if __name__ == "__main__" :
  def test_WFPeaks() :
    print(50*'_')
    kwargs = {'numchs'   : 5,
              'numhits'  : 16,
             }
    cfdpars= {'cfd_base'       :  0.,
              'cfd_thr'        : -0.05,
              'cfd_cfr'        :  0.85,
              'cfd_deadtime'   :  10.0,
              'cfd_leadingedge':  True,
              'cfd_ioffsetbeg' :  1000,
              'cfd_ioffsetend' :  2000,
              'cfd_wfbinbeg'   :  6000,
              'cfd_wfbinend'   : 22000,
             }
    kwargs.update(cfdpars)

    o = WFPeaks(**kwargs)

#----------

  def usage(tname):
    s = '\nUsage: python %s <test-number>' % sys.argv[0]
    if tname in ('0',)    : s+='\n 0 - test ALL'
    if tname in ('0','1') : s+='\n 1 - test_WFPeaks()'
    return s

#----------

if __name__ == "__main__" :
    import numpy as np; global np
    import sys; global sys
    fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s' # '%(message)s'
    logging.basicConfig(format=fmt, datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)

    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    s = 'End of Test %s' % tname
    print('%s' % usage('0')) # tname
    print(50*'_', '\nTest %s' % tname)
    if tname in ('0','1') : test_WFPeaks()
    if tname in ('0','2') : test_WFPeaks()
    print('%s' % usage(tname))
    print('%s' % usage('0')) # tname
    sys.exit(s)

#----------

