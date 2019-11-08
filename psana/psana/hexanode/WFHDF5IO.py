#----------

"""
Class :py:class:`WFHDF5IO` - HDF5 I/O for waveforms
===============================================================

Usage ::

    # Import and create object
    #--------------------------
    from psana.hexanode.WFHDF5IO import WFHDF5IO

    # **kwargs - contains peak-finder parameters, see test_WFHDF5IO() below.
    oh5 = WFHDF5IO(wfpeaks, **kwargs)

    # Access methods
    #----------------
    # wfs,wts - input arrays of waveforms and sample times, shape=(nchannels, nsamples)
    # get all-in-one:


Created on 2019-11-07 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

import h5py
from time import time
import psana.pyalgos.generic.Utils as gu
#import numpy as np
#from psana.pyalgos.generic.NDArrUtils import print_ndarr

#----------

class WFHDF5IO :

    def __init__(self, wfpeaks, **kwargs) :
        """
        """
        logger.info(gu.str_kwargs(kwargs, title='WFHDF5IO input parameters:'))
        self._of = None
        self._if = None

        self._wfpeaks        = wfpeaks
        self.NUM_CHANNELS    = wfpeaks.NUM_CHANNELS # kwargs.get('numchs',5) #  (int) - number 
        self.NUM_HITS        = wfpeaks.NUM_HITS     # kwargs.get('numhits',16) # (int) - maximal
        self.RUN             = kwargs.get('run', 0)
        self.EXP             = kwargs.get('exp', 'test1234')
        self.TDC_RESOLUTION  = kwargs.get('tdc_resolution', 0.250) # ns
        self._size_increment = kwargs.get('size_increment', 4096)
        self._start_time_sec = time()


    def open_output_h5file(self, fname='./test.h5') :
        self._nev = 0
        self._nevmax = self._size_increment
        self._of = h5py.File(fname,'w')
        dtype_str = h5py.special_dtype(vlen=str)
        self.h5ds_run           = self._of.create_dataset('run',          (1,), dtype=dtype_str)
        self.h5ds_exp           = self._of.create_dataset('experiment',   (1,), dtype=dtype_str)
        self.h5ds_start_time    = self._of.create_dataset('start_time',   (1,), dtype='f')
        self.h5ds_stop_time     = self._of.create_dataset('stop_time',    (1,), dtype='f')
        self.h5ds_proc_time     = self._of.create_dataset('proc_time',    (1,), dtype='f')
        self.h5ds_tdc_res_ns    = self._of.create_dataset('tdc_res_ns',   (1,), dtype='f')
        self.h5ds_nevents       = self._of.create_dataset('nevents',      (1,), dtype='i')
        self.h5ds_event_number  = self._of.create_dataset('event_number', (self._nevmax,), dtype='i', maxshape=(None,))
        self.h5ds_event_time    = self._of.create_dataset('event_time',   (self._nevmax,), dtype='f', maxshape=(None,))
        self.h5ds_nhits = self._of.create_dataset('nhits', (self._nevmax, self.NUM_CHANNELS), dtype='i',\
                                                            maxshape=(None, self.NUM_CHANNELS))
        self.h5ds_tdcns = self._of.create_dataset('tdcns', (self._nevmax, self.NUM_CHANNELS, self.NUM_HITS), dtype='f',\
                                                            maxshape=(None, self.NUM_CHANNELS, self.NUM_HITS))

    def close_output_h5file(self) :
        if self._of is None : return

        self._stop_time_sec = time()
        self._resize_h5_datasets(self._nev)
        self.h5ds_start_time[0] = self._start_time_sec
        self.h5ds_stop_time[0]  = self._stop_time_sec
        self.h5ds_proc_time[0]  = self._stop_time_sec - self._start_time_sec
        self.h5ds_nevents[0]    = self._nev
        self.h5ds_tdc_res_ns[0] = self.TDC_RESOLUTION
        self.h5ds_run[0]        = self.RUN
        self.h5ds_exp[0]        = self.EXP
        logger.info('Close output file: %s' % self._of.filename)
        self._of.close()
        self._of = None


    def _resize_h5_datasets(self, size=None) :
        self._nevmax = self._nevmax + self._size_increment if size is None else size
        self._of.flush()
        self.h5ds_nhits       .resize(self._nevmax, axis=0)   # or dset.resize((20,1024))
        self.h5ds_tdcns       .resize(self._nevmax, axis=0)   # or dset.resize((20,1024))
        self.h5ds_event_number.resize(self._nevmax, axis=0) 
        self.h5ds_event_time  .resize(self._nevmax, axis=0) 


    def add_event_to_h5file(self) :
        assert (self._wfpeaks._wfs_old is not None),\
               "waveforms need to be processed before calling add_event_to_h5file()"
        i = self._nev
        self.h5ds_nhits       [i] = self._wfpeaks._number_of_hits
        self.h5ds_tdcns       [i] = self._wfpeaks._pkt_ns
        self.h5ds_event_number[i] = i
        self.h5ds_event_time  [i] = time()
        self._nev += 1
        if self._nev > self._nevmax : self._resize_h5_datasets()


    def open_input_h5file(self, fname='./test.h5') :
        self._nev = 0
        self._if = h5py.File(fname,'r')
        self.h5ds_nhits      = self._if['nhits']
        self.h5ds_tdcns      = self._if['tdcns']
        self.h5ds_nevents    = self._if['nevents'][0]
        self.EXP             = self._if['experiment'][0]
        self.RUN             = self._if['run'][0]
        self.TDC_RESOLUTION  = self._if['tdc_res_ns'][0]
        self._start_time_sec = self._if['start_time'][0]
        self._stop_time_sec  = self._if['stop_time'][0]
        logger.info('File %s has %d records' % (fname, self.h5ds_nevents))


    def close_input_h5file(self, pbits=0) :
        if self._if is None : return
        logger.info('Close input file: %s' % self._if.filename)
        self._if.close()
        self._if = None


    def fetch_event_data_from_h5file(self, nev=None) :
        i = nev if nev is not None else self._nev
        if i>self.h5ds_nevents-1 : 
             return False
        self._number_of_hits = self.h5ds_nhits[i]
        self._tdc_ns         = self.h5ds_tdcns[i]
        self._nev += 1
        return True


    def __del__(self) :
        logger.debug('In WFHDF5IO.__del__')
        self.close_output_h5file()
        self.close_input_h5file()

#----------

