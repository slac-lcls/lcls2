"""
Class :py:class:`HexDataIO` a set of methods to access psana data of hexanode detector
======================================================================================

HexDataIO - a set of methods to access data, resembling hexanode/src/LMF_IO.cpp

Usage ::

    Create object and access methods
    --------------------------------
    from expmon.HexDataIO import HexDataIO

    o = HexDataIO(srcchs={'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)}, numchs=7, numhits=16)

    o.open_input_dataset('exp=xpptut15:run=390')

    # OR:
    o.open_input_h5file(ofname='./test.h5')

    # OR:
    ds = psana.MPIDataSource('exp=xpptut15:run=390')
    o.use_psana_dataset(ds, pbits=1022)


    status = o.read_next_event()           # gets next event from dataset, returns True if event is available

    nhits   = o.get_number_of_hits_array() # number of hits per channel, shape=(NUM_CHANNELS,)
    tdc_ns  = o.get_tdc_data_array()       # array of hit time [ns], shape=(NUM_CHANNELS, NUM_HITS)
    tdc_ind = o.get_tdc_index_array()      # array of hit index, shape=(NUM_CHANNELS, NUM_HITS)
    wf      = o.get_wf(channel=1)          # per channel array of intensities
    wt      = o.get_wt(channel=1)          # per channel array of times [s]
    nch     = o.get_number_of_channels()   # returns a number of channels
    t_ns    = o.tdc_resolution()           # returns TDC bin width in ns

    t_sec   = o.start_time_sec()           # returns (loatr) run start time
    t_sec   = o.stop_time_sec()            # returns (float) run start time
    tstamp  = o.start_time()               # returns (str) time stamp like '2017-10-23T17:00:00'
    tstamp  = o.stop_time()                # returns (str) time stamp like '2017-10-23T17:00:00'

    o.open_output_h5file(ofname='./test.h5')
    o.close_output_h5file()
    o.close_input_h5file()
    o.add_event_to_h5file()

    o.print_tdc_data()
    o.print_times()

Created on 2017-10-23 by Mikhail Dubrovin
Adapted HDF5-part for LCLS2 on 2019-09-25 by Mikhail Dubrovin
TODO: access to data through the detector interface 
"""
#------------------------------

from time import time
import numpy as np
#import pyimgalgos.GlobalUtils as gu
import psana.pyalgos.generic.Utils as gu

#------------------------------
from pypsalg import find_edges

from PSCalib.DCUtils import env_time, evt_time, evt_fiducials, str_tstamp
from expmon.PSUtils import exp_run_from_dsname # event_time, 

#------------------------------

def do_print(nev) :
    return nev<10\
       or (nev<50 and (not nev%10))\
       or (nev<500 and (not nev%100))\
       or not nev%1000

#------------------------------    

class HexDataIO :

    def __init__(self, **kwargs) :
        """Parameters

           - srcchs={'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)}
             (dict) - dictionary of pairs (source:list-of-channels)
             where list-of-channels should be for u1,u2,v1,v2,w1,w2,mcp signals
           - numchs=7 (int) - total number of channels in sources 
           - numhits=16 (int) - maximal number of hits in waveforms
        """
        self._tdc_resolution = None
        self._env = None
        self._events = None
        self._oh5file = None
        self._ih5file = None
        self.ds = None
        self._evnum =-1
        self._evt = None
        self.t0_sec = time()
        self._size_increment = 4096

        self.set_wf_hit_finder_parameters(**kwargs) 
        self._set_parameters(**kwargs)
        self._init_arrays()


    def _set_parameters(self, **kwargs) :
        """Sets parameters from kwargs
        """
        self.SRCCHS       = kwargs.get('srcchs', {'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)})
        self.NUM_CHANNELS = kwargs.get('numchs',7) #  (int) - number 
        self.NUM_HITS     = kwargs.get('numhits',16) # (int) - maximal
        self.DSNAME       = kwargs.get('dsname', 'exp=xpptut15:run=390:smd')


    def _init_arrays(self) :
        self._error_flag = 0
        self._event_is_processed = False
        self._number_of_hits = np.zeros((self.NUM_CHANNELS),    dtype=np.int)
        self._tdc_ns  = np.zeros((self.NUM_CHANNELS, self.NUM_HITS), dtype=np.double)
        self._tdc_ind = np.zeros((self.NUM_CHANNELS, self.NUM_HITS), dtype=np.int)
        self._dic_wf = {}
        self._dic_wt = {}


    def open_input_data(self, **kwargs) :
        dsname   = kwargs.get('dsname', 'exp=xpptut15:run=390:smd')
        do_mpids = kwargs.get('do_mpids', False)
        pbits    = kwargs.get('pbits', 0)

        if '.h5' in dsname : 
            self.open_input_h5file(dsname)
        else :
            self.open_input_dataset(dsname, pbits, do_mpids)
            # already done in __init__
            #self.set_wf_hit_finder_parameters(**kwargs)
            #self.print_wf_hit_finder_parameters()


    def open_input_dataset(self, dsname='exp=xpptut15:run=390:smd', pbits=1022, do_mpids=False) :
        import psana
        ds = psana.MPIDataSource(dsname) if do_mpids else\
             psana.DataSource(dsname)
        self.use_psana_dataset(ds, pbits)


    def use_psana_dataset(self, ds, pbits=1022) :
        """This method is used directly for external defenition of psana dataset and regular event loop.
        """
        import Detector.PyDataAccess as pda
        from Detector.WFDetector import WFDetector

        self.ds = ds
        self._env = self.ds.env()

        self._events = self.ds.events()
        self._pbits = pbits

        #nrun = evt.run()
        #evt = ds.events().next()
        #for key in evt.keys() : print key

        self.sources  = self.SRCCHS.keys()
        self.channels = self.SRCCHS.values()

        self.wfdets = [WFDetector(src, self._env, pbits) for src in self.sources]
        self.srcs_dets_channels = zip(self.sources, self.wfdets, self.channels)

        if pbits & 1 :
            for wfd in self.wfdets :
                wfd.print_attributes()
            self.print_wf_hit_finder_parameters()

        co = pda.get_acqiris_config_object(self._env, self.wfdets[0].source)
        self._tdc_resolution = co.horiz().sampInterval() * 1e9 # sec -> ns

        self._start_time_sec = env_time(self._env)
        self._stop_time_sec = self._start_time_sec + 1234

        self._exp, self._run = exp_run_from_dsname(self.DSNAME)


    def open_output_h5file(self, fname='./test.h5') :
        import h5py
        self._nevmax = self._size_increment
        self._oh5file = h5py.File(fname,'w')
        dtype_str = h5py.special_dtype(vlen=str)
        self.h5ds_run           = self._oh5file.create_dataset('run',          (1,), dtype=dtype_str)
        self.h5ds_exp           = self._oh5file.create_dataset('experiment',   (1,), dtype=dtype_str)
        self.h5ds_start_time    = self._oh5file.create_dataset('start_time',   (1,), dtype='f')
        self.h5ds_stop_time     = self._oh5file.create_dataset('stop_time',    (1,), dtype='f')
        self.h5ds_tdc_res_ns    = self._oh5file.create_dataset('tdc_res_ns',   (1,), dtype='f')
        self.h5ds_proc_time_sec = self._oh5file.create_dataset('proc_time_sec',(1,), dtype='f')
        self.h5ds_nevents       = self._oh5file.create_dataset('nevents',      (1,), dtype='i')
        self.h5ds_event_number  = self._oh5file.create_dataset('event_number', (self._nevmax,), dtype='i', maxshape=(None,))
        self.h5ds_event_time    = self._oh5file.create_dataset('event_time',   (self._nevmax,), dtype='f', maxshape=(None,))
        self.h5ds_fiducials     = self._oh5file.create_dataset('fiducials',    (self._nevmax,), dtype='i', maxshape=(None,))
        self.h5ds_nhits = self._oh5file.create_dataset('nhits', (self._nevmax, self.NUM_CHANNELS), dtype='i', maxshape=(None, self.NUM_CHANNELS))
        self.h5ds_tdcns = self._oh5file.create_dataset('tdcns', (self._nevmax, self.NUM_CHANNELS, self.NUM_HITS), dtype='f',\
                                                               maxshape=(None, self.NUM_CHANNELS, self.NUM_HITS))


    def close_output_h5file(self, pbits=0) :
        if self._oh5file is None : return
        start_time_sec = env_time(self._env)
        nevents = self.nevents_processed()
        self._resize_h5_datasets(nevents)
        self.h5ds_start_time[0] = self.start_time_sec()
        self.h5ds_stop_time[0]  = self.stop_time_sec()
        self.h5ds_nevents[0]    = nevents
        self.h5ds_tdc_res_ns[0] = self.tdc_resolution()
        self.h5ds_run[0]        = self._run
        self.h5ds_exp[0]        = self._exp
        self.h5ds_proc_time_sec[0] = time() - self.t0_sec

        if self._oh5file is None : return
        if pbits : print('Close output file: %s' % self._oh5file.filename)
        self._oh5file.close()
        self._oh5file = None


    def _resize_h5_datasets(self, size=None) :
        self._nevmax = self._nevmax + self._size_increment if size is None else size
        self._oh5file.flush()
        self.h5ds_nhits.resize(self._nevmax, axis=0)   # or dset.resize((20,1024))
        self.h5ds_tdcns.resize(self._nevmax, axis=0)   # or dset.resize((20,1024))
        self.h5ds_event_number.resize(self._nevmax, axis=0) 
        self.h5ds_event_time  .resize(self._nevmax, axis=0) 
        self.h5ds_fiducials   .resize(self._nevmax, axis=0)


    def add_event_to_h5file(self) :
        i = self.event_number()
        if i >= self._nevmax : self._resize_h5_datasets()
        self._proc_waveforms()

        self.h5ds_nhits[i] = self._number_of_hits
        self.h5ds_tdcns[i] = self._tdc_ns
        self.h5ds_event_number[i] = i
        self.h5ds_event_time[i] = evt_time(self._evt)
        self.h5ds_fiducials[i] = evt_fiducials(self._evt)

        #gu.print_ndarr(self._number_of_hits, '  _number_of_hits')
        #gu.print_ndarr(self._tdc_ns, '  _tdc_ns')
        #self.h5ds_nhits.attrs['events'] = i


    def open_input_h5file(self, fname='./test.h5') :
        import h5py
        self._ih5file = h5py.File(fname,'r')
        self.h5ds_nhits = self._ih5file['nhits']
        self.h5ds_tdcns = self._ih5file['tdcns']
        self.h5ds_nevents = self._ih5file['nevents'][0]
        print('File %s has %d records' % (fname, self.h5ds_nevents))
        #self.h5ds_nevents = self.h5ds_nhits.attrs['events']
        self._exp = self._ih5file['experiment'][0]
        self._run = self._ih5file['run'][0]
        self._tdc_resolution = self._ih5file['tdc_res_ns'][0]
        self._start_time_sec = self._ih5file['start_time'][0]
        self._stop_time_sec  = self._ih5file['stop_time'][0]


    def close_input_h5file(self, pbits=0) :
        if self._ih5file is None : return
        if pbits : print('Close input file: %s' % self._ih5file.filename)
        self._ih5file.close()
        self._ih5file = None


    def fetch_event_data_from_h5file(self) :
        i = self.event_number()
        if not (i<self.h5ds_nevents) : 
             return False
        self._number_of_hits = self.h5ds_nhits[i]
        self._tdc_ns         = self.h5ds_tdcns[i]
        return True


    def __del__(self) :
        self.close_output_h5file()
        self.close_input_h5file()


    def events(self) : 
        return self._events


    def env(self) : 
        return self._env


    def experiment(self) :
        return self._exp


    def run(self) :
        return self._run


    def runnum(self) :
        return int(self._run.lstrip('0'))


    def tdc_resolution(self) :
        return self._tdc_resolution


    def start_time_sec(self) :
        return self._start_time_sec


    def stop_time_sec(self) :
        return self._stop_time_sec


    def start_time(self) :
        """Returns (str) timestamp, e.g. '2017-10-23T17:00:00'
        """
        tsec = self.start_time_sec()
        return str_tstamp(fmt='%Y-%m-%dT%H:%M:%S', time_sec=int(tsec))


    def stop_time(self) :
        tsec = self.stop_time_sec()
        return str_tstamp(fmt='%Y-%m-%dT%H:%M:%S', time_sec=int(tsec))


    def proc_time_sec(self) :
        """ONLY For h5 input"""
        return self._ih5file['proc_time_sec'][0] if self._ih5file is not None else\
               None


    def number_of_events(self) :
        """ONLY For h5 input"""
        return self._ih5file['nevents'][0] if self._ih5file is not None else\
               self.nevents_processed()


    def get_number_of_channels(self) :
        return self.NUM_CHANNELS


    def set_next_event(self, evt, nevent=None) :
        """The same as read_next_event, but for external event loop
        """
        if evt==self._evt : return  False # if call it twice

        if nevent is None : self._evnum += 1
        else              : self._evnum = nevent

        if evt is None : return False
        self._init_arrays()
        self._evt = evt
        return True


    def read_next_event(self) :
        self._evnum += 1

        if self._ih5file is not None :
            #if self._evt is None : 
            #    self._init_arrays()
            #    self._evt = self._events.next()
            return self.fetch_event_data_from_h5file()

        self._init_arrays()
        self._evt = self._events.next()
        return self._evt is not None


    def event_number(self) :
        return self._evnum


    def get_event_number(self) :
        return self._evnum


    def nevents_processed(self) :
        return self.event_number() + 1


    def get_number_of_hits_array(self, arr=None) :
        self._proc_waveforms()
        if arr is not None : arr[:] = self._number_of_hits[:]
        return self._number_of_hits


    def get_tdc_data_array(self, arr=None) :
        self._proc_waveforms()
        if arr is not None : arr[:] = self._tdc_ns[:]
        return self._tdc_ns


    def get_tdc_index_array(self) :
        self._proc_waveforms()
        return self._tdc_ind


    def get_wf(self, channel=0) :
        self._proc_waveforms()
        #wf = self._dic_wf.get(channel, None)
        #gu.print_ndarr(wf, '    XXX waveform')
        return self._dic_wf.get(channel,None)


    def get_wt(self, channel=0) :
        self._proc_waveforms()
        return self._dic_wt.get(channel, None)


    def _proc_waveforms(self) :
        if self._ih5file is not None : return
        if self._event_is_processed : return
        self.proc_waveforms_for_evt(self._evt)
        self._event_is_processed = True


    def set_wf_hit_finder_parameters(self, **kwargs) :
        self.BASE        = kwargs.get('cfd_base',        0.)
        self.THR         = kwargs.get('cfd_thr',        -0.04)
        self.CFR         = kwargs.get('cfd_cfr',         0.9)
        self.DEADTIME    = kwargs.get('cfd_deadtime',    5.0)
        self.LEADINGEDGE = kwargs.get('cfd_leadingedge', True)
        self.IOFFSETBEG  = kwargs.get('cfd_ioffsetbeg',  0)
        self.IOFFSETEND  = kwargs.get('cfd_ioffsetend',  1000)
        self.WFBINBEG    = kwargs.get('cfd_wfbinbeg',    0)
        self.WFBINEND    = kwargs.get('cfd_wfbinend',    40000)


    def print_wf_hit_finder_parameters(self) :
        msg = '%s\nIn HexDataIO.print_wf_hit_finder_parameters' % (50*'_')\
            + '\nCFD parameters:'\
            + '\n  cfd_base         %.1f' % self.BASE\
            + '\n  cfd_thr          %.3f' % self.THR\
            + '\n  cfd_cfr          %.3f' % self.CFR\
            + '\n  cfd_deadtime     %.3f' % self.DEADTIME\
            + '\n  cfd_leadingedge    %s' % self.LEADINGEDGE\
            + '\n  cfd_ioffsetbeg     %d' % self.IOFFSETBEG\
            + '\n  cfd_ioffsetend     %d' % self.IOFFSETEND\
            + '\n  cfd_wfbinbeg       %d' % self.WFBINBEG\
            + '\n  cfd_wfbinend       %d' % self.WFBINEND\
            + '\n%s' % (50*'_')
        print(msg)


    def proc_waveforms_for_evt(self, evt) :
        ch_tdc = -1
        for src, wfd, channels in self.srcs_dets_channels :
            res = wfd.raw(evt)
            if res is None : continue
            wf,wt = res
            #print('XXX src, wfd, channels', src, wfd, channels)
            for ch in channels :
                #print('  XXX ch:', ch,)
                ch_tdc+=1
                if ch_tdc == self.NUM_CHANNELS :
                    raise IOError('HexDataIO._proc_waveforms: input tdc_ns shape=%s ' % str(tdc_ns.shape)\
                                  +' does not have enough rows for quad-/hex-anode channels')

                offset = wf[ch,self.IOFFSETBEG:self.IOFFSETEND].mean()
                wfch   = wf[ch,self.WFBINBEG:self.WFBINEND]
                wfch  -= offset
                self._dic_wf[ch_tdc] = wfch
                self._dic_wt[ch_tdc] = wtch = wt[ch,self.WFBINBEG:self.WFBINEND] * 1e9 # sec -> ns

                edges = find_edges(wfch, self.BASE, self.THR, self.CFR, self.DEADTIME, self.LEADINGEDGE)

                nedges = len(edges)
                #print(' ch_tdc:', ch_tdc, ' nedges:', nedges)

                if nedges >= self.NUM_HITS :
                    if self._pbits :
                        msg = 'HexDataIO._proc_waveforms: input tdc_ns shape=%s ' % str(self._tdc_ns.shape)\
                            + ' does not have enough columns for %d time records,' % nedges\
                            + '\nWARNING: NUMBER OF SIGNAL TIME RECORDS TRANCATED'
                        print(msg)
                    continue

                nhits = min(self.NUM_HITS, nedges) 
                self._number_of_hits[ch_tdc] = nhits

                for i in range(nhits):
                    amp,ind = edges[i]
                    self._tdc_ind[ch_tdc, i] = int(ind)
                    self._tdc_ns [ch_tdc, i] = wtch[int(ind)]


    def print_tdc_data(self) :
        for src, wfd, channels in self.srcs_dets_channels :
            print('source: %s channels: %s' % (src, str(channels)))
            res = wfd.raw(self._evt)
            if res is None : continue
            wf,wt = res
            gu.print_ndarr(wf, '    waveform')
            gu.print_ndarr(wt, '    wavetime')


    def print_times(self) :
        arr_nhits   = self.get_number_of_hits_array()
        arr_tdc_ns  = self.get_tdc_data_array()
        arr_tdc_ind = self.get_tdc_index_array()
        print('HexDataIO.print_times - event waveform times')
        print('Ch.#   Nhits   Index/Time[ns]')
        for ch in range(arr_nhits.size) :
            nhits = arr_nhits[ch]
            print('%4d   %4d:' % (ch, nhits),)
            for ihit in range(nhits) :
                print(' %9d' % (arr_tdc_ind[ch, ihit]),)
            print('\n              ',)
            for ihit in range(nhits) :
                print('  %8.1f' % (arr_tdc_ns[ch, ihit]),)
            print('')


    def error_flag(self) :
        return self._error_flag


    def get_error_text(self, error_flag) :
        #self._error_flag = 0
        return 'no-error'

#------------------------------
#------------------------------
#------------------------------

    def calib_dir(self) :
        import PSCalib.GlobalUtils as gu; global gu
        #ctype = 'hex_table'
        #ctype = 'hex_config'        
        #run = self.run()
        exp = self.experiment()
        return gu.calib_dir_for_exp(exp)


    def calib_src(self) :
        return self.SRCCHS.keys()[0]


    def calib_group(self) :
        src = self.calib_src()
        dettype = gu.det_type_from_source(src)
        return gu.dic_det_type_to_calib_group[dettype]


    def calibtype_dir(self) :
        cdir = self.calib_dir()
        grp  = self.calib_group()
        src  = self.calib_src()
        return '%s/%s/%s' % (cdir, grp, src)


    def find_calib_file(self, type='hex_config', run=None, pbits=1) :
        import PSCalib.CalibFileFinder as cff
        cdir = self.calib_dir()
        src  = self.calib_src()
        rnum = run if run is not None else self.runnum()
        return cff.find_calib_file(cdir, src, type, rnum, pbits=pbits)


    def make_calib_file_path(self, type='hex_config', run=None, pbits=1) : 
        import PSCalib.CalibFileFinder as cff
        cdir = self.calib_dir()
        src  = self.calib_src()
        rnum = run if run is not None else self.runnum()
        return cff.make_calib_file_name(cdir, src, type, rnum, run_end=None, pbits=pbits)

#------------------------------
#------------ TEST ------------
#------------------------------

def quaddataio(pbits=1022) :
    o = HexDataIO(srcchs={'AmoEndstation.0:Acqiris.1':(2,3,4,5,6)}, numchs=5, numhits=16)
    o.open_input_dataset('/reg/g/psdm/detector/data_test/hdf5/amox27716-r0100-e060000-single-node.h5', pbits)
    #o.open_input_dataset('exp=amox27716:run=100', pbits)
    return o

#------------------------------

def hexdataio(pbits=1022) :
    o = HexDataIO(srcchs={'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)}, numchs=7, numhits=16)
    o.open_input_dataset('exp=xpptut15:run=390', pbits)
    return o

#------------------------------

def test_quaddataio(EVENTS=10) :
    o = quaddataio()
    while o.get_event_number() < EVENTS :
        print('%s\nEvent %d' % (80*'_', o.get_event_number()))
        o.read_next_event()
        o.print_tdc_data()
        nhits = o.get_number_of_hits_array()
        gu.print_ndarr(nhits, '    nhits', first=0, last=7)
        print('\nTDC resolution [ns]: %.3f' % o.tdc_resolution())
        o.print_times()

#------------------------------

def test_hexdataio(EVENTS=10) :
    o = hexdataio()
    while o.get_event_number() < EVENTS :
        print('%s\nEvent %d' % (80*'_', o.get_event_number()))
        o.read_next_event()
        o.print_tdc_data()
        nhits = o.get_number_of_hits_array()
        gu.print_ndarr(nhits, '    nhits', first=0, last=7)
        print('\nTDC resolution [ns]: %.3f' % o.tdc_resolution())
        o.print_times()

#------------------------------

def draw_times(ax, wf, wt, nhits, hit_inds) :
    print('nhits:%2d'%nhits,)
    for i in range(nhits) :
        hi = hit_inds[i]
        ti = wt[hi] # hit time
        ai = wf[hi] # hit intensity
        print(' %.1f' % ti,)
        gg.drawLine(ax, (ti,ti), (ai,-ai), s=10, linewidth=1, color='k')
    print('')

#------------------------------

def test_hexdataio_graph(EVENTS=10, EVSKIP=0) :

    import pyimgalgos.Graphics       as gr; global gr
    import pyimgalgos.GlobalGraphics as gg; global gg

    fig = gr.figure(figsize=(15,15), title='Image')

    nchans = 7
    dy = 1./nchans

    lw = 1
    w = 0.87
    h = dy - 0.04
    x0, y0 = 0.07, 0.03

    gfmt = ('b-', 'r-', 'g-', 'k-', 'm-', 'y-', 'c-', )
    ylab = ('Y1', 'Y2', 'Z1', 'Z2', 'X1', 'X2', 'MCP', )
    ax = [gr.add_axes(fig, axwin=(x0, y0 + i*dy, w, h)) for i in range(nchans)]

    o = hexdataio(pbits=0)
    evnum = 0
    while evnum < EVENTS :
        evnum = o.get_event_number()
        print('%s\nEvent %d' % (80*'_', evnum))
        o.read_next_event()
        if evnum < EVSKIP : continue

        o.print_times()

        nhits = o.get_number_of_hits_array()
        hit_inds = o.get_tdc_index_array()
        gu.print_ndarr(nhits, 'nhits', first=0, last=7)

        for c in range(len(nhits)) :
            gu.print_ndarr(o.get_wf(c), 'ch:%2d wf'%c, first=0, last=5)
        gu.print_ndarr(o.get_wt(0), 'ch:%2d wt'%c, first=0, last=4)

        gr.set_win_title(fig, titwin='Event: %d' % o.get_event_number())

        for c in range(nchans) :
            ax[c].clear()
            ax[c].set_xlim((2500,4500)) # [ns]
            ax[c].set_ylabel(ylab[c], fontsize=14)
 
            wt, wf = o.get_wt(c), o.get_wf(c)
            if None in (wt, wf) : continue
            wtsel = wt[:-1]
            wfsel = wf[:-1]

            ax[c].plot(wtsel, wfsel, gfmt[c], linewidth=lw)
            gg.drawLine(ax[c], ax[c].get_xlim(), (o.THR, o.THR), s=10, linewidth=1, color='k')
            draw_times(ax[c], wfsel, wtsel, nhits[c], hit_inds[c,:])
        gr.draw_fig(fig)
        gr.show(mode='non-hold')
    gr.show()
    o.print_wf_hit_finder_parameters()

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    import numpy as np; global np
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print(50*'_', '\nTest %s' % tname)
    if   tname == '1' : test_quaddataio()
    elif tname == '2' : test_hexdataio()
    elif tname == '3' : test_hexdataio_graph()
    else : print('Not-recognized test name: %s' % tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
#------------------------------
