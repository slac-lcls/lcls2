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
    pkinds, pkvals = peaks.peak_indexes_values(wfs, wts) # for V4

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
from ndarray import wfpkfinder_cfd # from psana.pycalgos
from psana.hexanode.WFUtils import peak_finder_v2, peak_finder_v3
from psana.hexanode.PyCFD import PyCFD

#----------

class WFPeaks :

    def __init__(self, **kwargs) :
        """Waveform peak finder wrapper.
           - wf digitizer channels (0,1,2,3,4) should be ordered for u1,u2,v1,v2[,w1,w2],mcp, respectively
        """
        logger.debug(gu.str_kwargs(kwargs, title='WFPeaks input parameters:'))
        self.set_wf_peak_finder_parameters(**kwargs)

        self._wfs_old = None

        self.tbins = None # need it in V4 to convert _pktsec to _pkinds and _pkvals

#----------

    def set_wf_peak_finder_parameters(self, **kwargs) :

        self.NUM_CHANNELS= kwargs.get('numchs',  5)
        self.NUM_HITS    = kwargs.get('numhits',16)
        self.VERSION     = kwargs.get('version', 1)

        if True :
            self.BASE        = kwargs.get('cfd_base',          0.)
            self.THR         = kwargs.get('cfd_thr',        -0.05)
            self.CFR         = kwargs.get('cfd_cfr',         0.85)
            self.DEADTIME    = kwargs.get('cfd_deadtime',    10.0)
            self.LEADINGEDGE = kwargs.get('cfd_leadingedge', True)
            self.IOFFSETBEG  = kwargs.get('cfd_ioffsetbeg',  1000)
            self.IOFFSETEND  = kwargs.get('cfd_ioffsetend',  2000)
            self.WFBINBEG    = kwargs.get('cfd_wfbinbeg',    6000)
            self.WFBINEND    = kwargs.get('cfd_wfbinend',   30000)

        if self.VERSION == 2 :
            self.SIGMABINS   = kwargs.get('pf2_sigmabins',      3)
            self.NSTDTHR     = kwargs.get('pf2_nstdthr',       -5)
            self.DEADBINS    = kwargs.get('pf2_deadbins',      10)
            self.IOFFSETBEG  = kwargs.get('pf2_ioffsetbeg',  1000)
            self.IOFFSETEND  = kwargs.get('pf2_ioffsetend',  2000)
            self.WFBINBEG    = kwargs.get('pf2_wfbinbeg',    6000)
            self.WFBINEND    = kwargs.get('pf2_wfbinend',   30000)

        if self.VERSION == 3 :
            self.SIGMABINS   = kwargs.get('pf3_sigmabins',      3)
            self.BASEBINS    = kwargs.get('pf3_basebins',     100)
            self.NSTDTHR     = kwargs.get('pf3_nstdthr',        5)
            self.GAPBINS     = kwargs.get('pf3_gapbins',      200)
            self.DEADBINS    = kwargs.get('pf3_deadbins',      10)
            # used in proc_waveforms
            self.IOFFSETBEG  = kwargs.get('pf3_ioffsetbeg',  1000)
            self.IOFFSETEND  = kwargs.get('pf3_ioffsetend',  2000)
            self.WFBINBEG    = kwargs.get('pf3_wfbinbeg',    6000)
            self.WFBINEND    = kwargs.get('pf3_wfbinend',   30000)
            
        if self.VERSION == 4 :
            self.paramsCFD = kwargs.get('paramsCFD', {})
            self.cnls = ['mcp','x1','x2','y1','y2']#for QUAD only
            self.cnls_map = {4:'mcp',0:'x1',1:'x2',2:'y1',3:'y2'}
            self.PyCFDs = {cnl:PyCFD(self.paramsCFD[cnl]) for cnl in self.cnls}
                
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

        if self.VERSION == 2 : std = wfs[:,self.IOFFSETBEG:self.IOFFSETEND].std(axis=1)


        self.wfsprep = wfs[:,self.WFBINBEG:self.WFBINEND] - offsets.reshape(-1, 1) # subtract wf-offset
        self.wtsprep = wts[:,self.WFBINBEG:self.WFBINEND] # sec

        for ch in range(self.NUM_CHANNELS) :

            wf = self.wfsprep[ch,:]
            wt = self.wtsprep[ch,:]

            npeaks = None
            if self.VERSION == 3 :
                npeaks, self.wfgi, self.wff, self.wfg, self.thrg, self.edges =\
                peak_finder_v3(wf, self.SIGMABINS, self.BASEBINS, self.NSTDTHR, self.GAPBINS, self.DEADBINS,\
                                        self._pkvals[ch,:], self._pkinds[ch,:])
            elif self.VERSION == 2 :
                self.THR = self.NSTDTHR*std[ch]
                npeaks = peak_finder_v2(wf, self.SIGMABINS, self.THR, self.DEADBINS,\
                                        self._pkvals[ch,:], self._pkinds[ch,:])
            elif self.VERSION == 4 :
                t_list = self.PyCFDs[self.cnls_map[ch]].CFD(wf,wt)
                npeaks = self._pkinds[ch,:].size if self._pkinds[ch,:].size<=len(t_list) else len(t_list)
                # need it in V4 to convert _pktsec to _pkinds and _pkvals
                if self.tbins is None :
                    from psana.pyalgos.generic.HBins import HBins
                    self.tbins = HBins(list(wt))

            else : # self.VERSION == 1
                npeaks = wfpkfinder_cfd(wf, self.BASE, self.THR, self.CFR, self.DEADTIME, self.LEADINGEDGE,\
                                        self._pkvals[ch,:], self._pkinds[ch,:])

            #print(' npeaks:', npeaks)
            #assert (npeaks<self.NUM_HITS), 'number of found peaks exceeds reserved array shape'
            if npeaks>=self.NUM_HITS : npeaks = self.NUM_HITS
            self._number_of_hits[ch] = npeaks
            if self.VERSION == 4 :
                self._pktsec[ch, :npeaks] = np.array(t_list)[:npeaks]
            else:
                self._pktsec[ch, :npeaks] = wt[self._pkinds[ch, :npeaks]] #sec

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

    def peak_indexes_values(self, wfs, wts) :
        """ added for V4 to convert _pktsec to _pkinds and _pkvals
        """
        self.proc_waveforms(wfs, wts)
        if self.VERSION == 4 :
          # This is SLOW for V4 graphics...
          for ch in range(self.NUM_CHANNELS) :
            npeaks = self._number_of_hits[ch]
            wf = self.wfsprep[ch,:]
            self._pkinds[ch, :npeaks] = self.tbins.bin_indexes(self._pktsec[ch, :npeaks])
            self._pkvals[ch, :npeaks] = wf[self._pkinds[ch, :npeaks]]
        return self._pkinds, self._pkvals

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
              'version'  : 1,
             }

    cfdpars= {'cfd_thr'        : -0.05,
              'cfd_cfr'        :  0.85,
              'cfd_deadtime'   :  10.0,
              'cfd_leadingedge':  True,
              'cfd_ioffsetbeg' :  1000,
              'cfd_ioffsetend' :  2000,
              'cfd_wfbinbeg'   :  6000,
              'cfd_wfbinend'   : 22000,
             }
    kwargs.update(cfdpars)

    pf2pars= {'pf2_sigmabins'  :     3,
              'pf2_nstdthr'    :    -5,
              'pf2_deadbins'   :    10,
              'pf2_ioffsetbeg' :  1000,
              'pf2_ioffsetend' :  2000,
              'pf2_wfbinbeg'   :  6000,
              'pf2_wfbinend'   : 22000,
             }
    kwargs.update(pf2pars)

    pf3pars= {'pf3_sigmabins'  :     3,
              'pf3_basebins'   :   100,
              'pf3_nstdthr'    :     5,
              'pf3_gapbins'    :   100,
              'pf3_deadbins'   :    20,
             }
    kwargs.update(pf3pars)

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

