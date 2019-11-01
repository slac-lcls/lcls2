"""
Class :py:class:`HexDataIOExt` - extension of HexDataIO for data processing with "3-line example"
=================================================================================================

Usage ::

    # Example 1 - dataset created outside object
    #-------------------------------------------
    from expmon.HexDataIOExt import HexDataIOExt  # Line 0 - import

    # dictionary of input parameters
    kwargs = {'command'  : 1,
              'srcchs'   : {'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)},
              'numchs'   : 7,
              'numhits'  : 16,
              'dsname'   : 'exp=xpptut15:run=390:smd',
              'evskip'   : 0,
              'events'   : 500,
              'verbose'  : False,
              'cfd_base'        :  0.  ,        
              'cfd_thr'         : -0.04,         
              'cfd_cfr'         :  0.9 ,         
              'cfd_deadtime'    :  5.0 ,    
              'cfd_leadingedge' :  True, 
              'cfd_ioffsetbeg'  :  0   ,  
              'cfd_ioffsetend'  :  1000, 
             }

    ds = psana.MPIDataSource(kwargs['dsname'])
    o = HexDataIOExt(ds, **kwargs)                # Line 1 - initialization

    for evt in ds.events() :
        if o.skip_event(evt) : continue           # Line 2 - loop control method passes evt to the object
        if o.event_number() > o.EVENTS : break

        #x, y, t = o.hits_xyt()                   # Line 3 - get arrays x, y, t of hits' coordinates and time
        o.print_hits()                            # prints x, y, time for all hits in the event

    o.print_summary() # print total number of events, processing time, frequency, etc


    # Example 2 - dataset created inside object
    #------------------------------------------
    from expmon.HexDataIOExt import HexDataIOExt # Line 0 - import

    # dictionary of input (non-default) parameters
    kwargs = {'srcchs' : {'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)},
              'numchs' : 7,
              'numhits': 16,
              'dsname' : 'exp=xpptut15:run=390:smd',
             }

    o = HexDataIOExt(**kwargs)             # Line 1 - initialization, dataset is defined inside the object 
    while o.read_next_event() :            # Line 2 - event loop
        o.print_sparsed_event_info()       # print sparsed event number and time consumption 
        if o.skip_event()       : continue # event loop control, skip events with zero hits
        if o.break_event_loop() : break    # event loop control, break ate teh end or when smth went wrong
        o.print_hits()                     # prints x, y, time, method for found in event hits
        x, y, t = o.hits_xyt()             # Line 3 - get arrays x, y, t of hits' coordinates and time

    #-------------
    # Make object
    #-------------
    o = HexDataIOExt(ds=None, **kwargs)

    # Event loop control methods
    #---------------------------
    status = o.read_next_event()
    stat = o.break_event_loop(
    stat = o.skip_event()
    stat = o.skip_event(evt) # pass evt for external dataset

    # Methods per event
    #-------------------
    o.print_sparsed_event_info()
    o.print_summary()
    o.print_hits()
    o.calib_command2() # for calibration with command 2
    o.calib_command3() # for calibration with command 3

    # Access sorted data info, hit coordinates, times, hit finding method
    #--------------------------------------------------------------------
    x, y, t = o.hits_xyt()
    x = o.hits_x()
    y = o.hits_y()
    t = o.hits_t()
    m = o.hits_method()

Created on 2017-12-14 by Mikhail Dubrovin.
"""
#------------------------------

import os
from math import sqrt
from expmon.HexDataIO import * # HexDataIO, do_print, etc.
import hexanode

#------------------------------

OSQRT3 = 1./sqrt(3.)

#------------------------------

def create_output_directory(prefix) :
    dirname = os.path.dirname(prefix)
    print 'Output directory: "%s"' % dirname
    if dirname in ('', './', None) : return
    from CalibManager.GlobalUtils import create_directory # , create_path, 
    #create_path(dirname, depth=2, mode=0775)
    create_directory(dirname, mode=0775)

#------------------------------

class HexDataIOExt(HexDataIO) :

    def __init__(self, ds=None, **kwargs) :
        """See kwargs description in expmon.HexDataIO
        """
        self._name = self.__class__.__name__
        print 'In %s.__init__' % self._name

        HexDataIO.__init__(self, **kwargs)

        DIO = self
        if ds is None :
            DIO.open_input_data(self.DSNAME, **kwargs)
        else :
            DIO.use_psana_dataset(ds, pbits=0377 if self.VERBOSE else 0)
    
        self._init_calib_and_sorter()

        self.t0_sec = self.t1_sec = time()


    def _set_parameters(self, **kwargs) :
        """Overrides HexDataIO._set_parameters
        """
        HexDataIO._set_parameters(self, **kwargs)

        # set in HexDataIO
        #self.SRCCHS       = kwargs.get('srcchs', {'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)})
        #self.DSNAME       = kwargs.get('dsname', 'exp=xpptut15:run=390:smd')
        #self.NUM_CHANNELS = kwargs.get('numchs', 7)
        #self.NUM_HITS     = kwargs.get('numhits', 16)

        self.COMMAND      = kwargs.get('command', 1)
        self.EVSKIP       = kwargs.get('evskip', 0)
        self.EVENTS       = kwargs.get('events', 1000000) + self.EVSKIP
        self.OFPREFIX     = kwargs.get('ofprefix','./figs-hexanode/plot')
        self.PLOT_HIS     = kwargs.get('plot_his', True)
        self.VERBOSE      = kwargs.get('verbose', False)
        self.calibtab     = kwargs.get('calibtab', None)
    
        print '%s: Input parameters:' % self._name
        for k,v in kwargs.iteritems() : print '%20s : %s' % (k,str(v))

        self.CTYPE_HEX_CONFIG = 'hex_config'
        self.CTYPE_HEX_TABLE  = 'hex_table'
    
        self.event_status = None

    
    def _init_calib_and_sorter(self) :
        print 'In %s._init_calib_and_sorter' % self._name
    
        DIO = self

        #=====================

        self.CALIBTAB = self.calibtab if self.calibtab is not None else\
                        DIO.find_calib_file(type=self.CTYPE_HEX_TABLE)
        self.CALIBCFG = DIO.find_calib_file(type=self.CTYPE_HEX_CONFIG)    

        #=====================
    
        print 'DIO experiment : %s' % DIO.experiment()
        print 'DIO run        : %s' % DIO.run()
        print 'DIO start time : %s' % DIO.start_time()
        print 'DIO stop time  : %s' % DIO.stop_time()
        print 'DIO tdc_resolution : %.3f' % DIO.tdc_resolution()
    
        print 'DIO calib_dir   : %s' % DIO.calib_dir()
        print 'DIO calib_src   : %s' % DIO.calib_src()
        print 'DIO calib_group : %s' % DIO.calib_group()
        print 'DIO ctype_dir   : %s' % DIO.calibtype_dir()
        print 'DIO find_calib_file config: %s' % self.CALIBCFG
        print 'DIO find_calib_file  table: %s' % self.CALIBTAB
    
        #=====================

        self.tdc_ns = np.zeros((self.NUM_CHANNELS, self.NUM_HITS), dtype=np.float64)
        self.number_of_hits = np.zeros((self.NUM_CHANNELS,), dtype=np.int32)
    
        # create the sorter
        self.sorter = hexanode.py_sort_class()
        sorter = self.sorter

        global command_cfg, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y

        status, command_cfg, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y=\
            hexanode.py_read_config_file(self.CALIBCFG, sorter)
        command = self.COMMAND # command_cfg
        print '%s: read_config_file' % self._name
        print 'status, COMMAND, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y:\n',\
               status, command, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y

        # The "command"-value is set in the first line of "sorter.txt" then substituted from input parameter.
        # 0 = only convert to new file format
        # 1 = sort and write new file 
        # 2 = calibrate fv, fw, w_offset
        # 3 = create calibration table files
     
        if not status :
            print "%s: WARNING: can't read config file %s" % (self._name, fname_cfg)
            sys.exit(0)

        #=====================
    
        print '%s: use_sum_correction %s' % (self._name, sorter.use_sum_correction)
        print '%s: use_pos_correction %s' % (self._name, sorter.use_pos_correction)
        if sorter is not None :
            if sorter.use_sum_correction or sorter.use_pos_correction :
                status = hexanode.py_read_calibration_tables(self.CALIBTAB, sorter)
    
        if command == -1 :
            print '%s: no config file was read. Nothing to do.' % self._name
            if sorter is not None : del sorter
            sys.exit(0)

        global Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp

        Cu1  = sorter.cu1
        Cu2  = sorter.cu2
        Cv1  = sorter.cv1 
        Cv2  = sorter.cv2 
        Cw1  = sorter.cw1 
        Cw2  = sorter.cw2
        Cmcp = sorter.cmcp
        print "Numeration of channels - u1:%i  u2:%i  v1:%i  v2:%i  w1:%i  w2:%i  mcp:%i"%\
              (Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp)
    
        #=====================
    
        print "%s: init sorter... " % self._name
    
        sorter.set_tdc_resolution_ns(DIO.tdc_resolution()) # 0.025
        sorter.set_tdc_array_row_length(self.NUM_HITS)
        sorter.set_count(self.number_of_hits)
        sorter.set_tdc_pointer(self.tdc_ns)
    
        #sorter.set_use_reflection_filter_on_u1(False) # Achim recommended False
        #sorter.set_use_reflection_filter_on_u2(False)
    
        if command >= 2 :
            sorter.create_scalefactors_calibrator(True,\
                                                  sorter.runtime_u,\
                                                  sorter.runtime_v,\
                                                  sorter.runtime_w, 0.78,\
                                                  sorter.fu, sorter.fv, sorter.fw)
    
        error_code = sorter.init_after_setting_parameters()
        if error_code :
            print "%s: sorter could not be initialized\n" % self._name
            error_text = sorter.get_error_text(error_code, 512)
            print '%s: Error %d: %s' % (self._name, error_code, error_text)
            sys.exit(0)


        print "Calibration factors:\n  f_U (mm/ns) =%f\n  f_V (mm/ns) =%f\n  f_W (mm/ns) =%f\n  Offset on layer W (ns) =%f\n"%\
              (2*sorter.fu, 2*sorter.fv, 2*sorter.fw, w_offset)
    
        create_output_directory(self.OFPREFIX)

        print "%s: ok for sorter initialization\n" % self._name
    
#------------------------------

    def set_next_event(self, evt, nevent=None) :
        """Re-implemented method from HexDataIO adding per event processing
        """
        status = HexDataIO.set_next_event(self, evt, nevent)
        if status : self.proc_event()
        return status


    def read_next_event(self) :
        """Re-implemented method from HexDataIO adding per event processing
        """
        status = HexDataIO.read_next_event(self)

        if not status : 
            self.event_status = 0 # to break the event loop
            return False

        self.proc_event()
        return True


    def proc_event(self) :
        """Process event data
        """
        DIO     = self
        sorter  = self.sorter
        command = self.COMMAND
        VERBOSE = self.VERBOSE

        tdc_ns         = self.tdc_ns
        number_of_hits = self.number_of_hits

        DIO.get_number_of_hits_array(number_of_hits)
        if DIO.error_flag() :
            error_text = DIO.get_error_text(DIO.error_flag())
            print "%s: DIO Error %d: %s" % (self._name, DIO.error_flag(), error_text)
            sys.exit(0)
        if VERBOSE : print '   number_of_hits_array', number_of_hits[:8]
    
        DIO.get_tdc_data_array(tdc_ns)    
        if DIO.error_flag() :
            error_text = DIO.get_error_text(DIO.error_flag())
            print "%s: DIO Error %d: %s" % (self._name, DIO.error_flag(), error_text)
            sys.exit(0)    
        if VERBOSE : print '   TDC data:\n', tdc_ns[0:8,0:5]
    
        # apply conversion to ns: DIO returns tdc_ns already in [ns], no correction needed
        # tdc_ns *= DIO.tdc_resolution()

        #=====================
        # Access to raw data before sorter: 
        #number_of_hits[Cu1]
        #number_of_hits[Cu2]
        #number_of_hits[Cv1]
        #number_of_hits[Cv2]
        #number_of_hits[Cw1]
        #number_of_hits[Cw2]
        #number_of_hits[Cmcp]

        #time_sum_u = tdc_ns[Cu1,0] + tdc_ns[Cu2,0] - 2*tdc_ns[Cmcp,0]
        #time_sum_v = tdc_ns[Cv1,0] + tdc_ns[Cv2,0] - 2*tdc_ns[Cmcp,0]
        #time_sum_w = tdc_ns[Cw1,0] + tdc_ns[Cw2,0] - 2*tdc_ns[Cmcp,0]
    
        #u_ns = tdc_ns[Cu1,0] - tdc_ns[Cu2,0]
        #v_ns = tdc_ns[Cv1,0] - tdc_ns[Cv2,0]
        #w_ns = tdc_ns[Cw1,0] - tdc_ns[Cw2,0]
    
        #u = u_ns * sorter.fu
        #v = v_ns * sorter.fv
        #w = (w_ns + w_offset) * sorter.fw
    
        #Xuv = u
        #Xuw = u
        #Xvw = v + w
        #Yuv = (u - 2*v)*OSQRT3
        #Yuw = (2*w - u)*OSQRT3
        #Yvw = (w - v)*OSQRT3
    
        #dX = Xuv - Xvw
        #dY = Yuv - Yvw
        #Deviation = sqrt(dX*dX + dY*dY)

        #=============================================================
        # Apply corrections to raw data

        if sorter.use_hex :        
            # shift the time sums to zero:
            sorter.shift_sums(+1, offset_sum_u, offset_sum_v, offset_sum_w)
            #shift layer w so that the middle lines of all layers intersect in one point:
            sorter.shift_layer_w(+1, w_offset)
        else :
            # shift the time sums to zero:
            sorter.shift_sums(+1, offset_sum_u, offset_sum_v)
    
            # shift all signals from the anode so that the center of the detector is at x=y=0:
            sorter.shift_position_origin(+1, pos_offset_x, pos_offset_y)
     
            sorter.feed_calibration_data(True, w_offset) # for calibration of fv, fw, w_offset and correction tables

        #=============================================================
        # Sort the TDC-Data and reconstruct missing signals and apply the sum- and NL-correction.
        # number_of_particles is the number of reconstructed particles

        self.number_of_particles = sorter.sort() if command == 1 else\
                                   sorter.run_without_sorting()

        #=============================================================

        if self.number_of_particles<1 : 
            self.event_status = 2 # to skip event
    
        #=====================
        # Access to calibrated data after sorter: 

        #DIO.get_tdc_data_array(tdc_ns)
        #time_sum_u_corr = tdc_ns[Cu1,0] + tdc_ns[Cu2,0] - 2*tdc_ns[Cmcp,0]
        #time_sum_v_corr = tdc_ns[Cv1,0] + tdc_ns[Cv2,0] - 2*tdc_ns[Cmcp,0]
        #time_sum_w_corr = tdc_ns[Cw1,0] + tdc_ns[Cw2,0] - 2*tdc_ns[Cmcp,0]
    
        #=====================
        self.event_status = 1 # good event
        #=====================

#------------------------------

    def calib_command2(self) :
        print '%s: calibrating detector... ' % self._name

        sorter.do_calibration()
        print "ok - after do_calibration"
        sfco = hexanode.py_scalefactors_calibration_class(sorter)
        if sfco :
            print "Good calibration factors are:\n  f_U =%f\n  f_V =%f\n  f_W =%f\n  Offset on layer W=%f\n"%\
                  (2*sorter.fu, 2*sfco.best_fv, 2*sfco.best_fw, sfco.best_w_offset)
    
            print 'CALIBRATION: These parameters and time sum offsets from histograms should be set in the file\n  %s' % self.CALIBCFG

#------------------------------

    def calib_command3(self) : # generate and print correction tables for sum- and position-correction
        print "%s: creating calibration tables..." % self._name
        DIO = self
        sorter = self.sorter
    
        self.CALIBTAB = self.calibtab if cself.alibtab is not None else\
                        DIO.make_calib_file_path(type=self.CTYPE_HEX_TABLE)
        status = hexanode.py_create_calibration_tables(self.CALIBTAB, self.sorter)
    
        print "CALIBRATION: finished creating calibration tables: %s status %s" % (self.CALIBTAB, status)

#------------------------------
    
    def print_sparsed_event_info(self) :
        evnum = self.event_number()
        if do_print(evnum) :
            t1 = time()
            print 'Event: %06d, dt(sec): %.3f' % (evnum, t1 - self.t1_sec)
            self.t1_sec = t1


    def print_summary(self) :
        dt = time() - self.t0_sec
        evnum = self.event_number() - 1
        dt_o_evnum = dt/evnum if evnum>0 else 0
        freq = evnum/dt if dt else 0
        print "%s: %d events processed, consumed time = %.6f sec or %.6f sec/event f=%.2f Hz\n"%\
              (self._name, evnum, dt, dt_o_evnum, freq)


    def print_hits(self) : 
        print "  %s: Event %5i number_of_particles: %i" % (self._name, self.event_number(), self.number_of_particles)
        for i in range(self.number_of_particles) :
            hco= hexanode.py_hit_class(self.sorter, i)
            print "    p:%2i x:%7.2f y:%7.2f t:%.2f met:%d" % (i, hco.x, hco.y, hco.time, hco.method)    


    def hits_xyt(self) : 
        x_hits = []
        y_hits = []
        t_hits = []
        for i in range(self.number_of_particles) :
            hco= hexanode.py_hit_class(self.sorter, i)
            x_hits.append(hco.x)
            y_hits.append(hco.y)
            t_hits.append(hco.time)
        return x_hits, y_hits, t_hits

    def hits_x(self) : 
        return [hexanode.py_hit_class(self.sorter, i).x for i in range(self.number_of_particles)]

    def hits_y(self) : 
        return [hexanode.py_hit_class(self.sorter, i).y for i in range(self.number_of_particles)]

    def hits_t(self) : 
        return [hexanode.py_hit_class(self.sorter, i).time for i in range(self.number_of_particles)]

    def hits_method(self) : 
        return [hexanode.py_hit_class(self.sorter, i).method for i in range(self.number_of_particles)]

#------------------------------

    def skip_event(self, evt=0, nevent=None) :
        """psana.Event can be None, so evt=0 is set to destinguish case w/o evt.
        """
        if evt!=0 : 
            # if parameter evt is passed (new version)
            status = self.set_next_event(evt, nevent)
            self.print_sparsed_event_info()     # print sparsed event number and time consumption 
            if evt is None :
                print '  Event: %4d WARNING: evt is None, rank: %d' % (self.event_number(), self.ds.rank)
                return True

            if not status : 
                return True

        return (self.event_number() < self.EVSKIP) or (self.event_status==2)


    def break_event_loop(self) : 
        return (self.event_number() > self.EVENTS) or (self.event_status==0)    

#------------------------------
#------------------------------
#------------------------------    
#------------------------------    
#------------------------------    

def test2_HexDataIOExt() :

    # Parameters for initialization of the data source, channels, number of events etc.
    kwargs = {'command'  : 1,
              'srcchs'   : {'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)},
              'numchs'   : 7,
              'numhits'  : 16,
              'dsname'   : 'exp=xpptut15:run=390:smd',
              'evskip'   : 0,
              'events'   : 500,
              'ofprefix' : './',
              'verbose'  : False,
             }

    o = HexDataIOExt(**kwargs)       # Line # 1

    while o.read_next_event() :      # Line # 2

        o.print_sparsed_event_info() # print sparsed event number and time consumption 

        if o.skip_event()       : continue # event loop control
        if o.break_event_loop() : break    # event loop control

        x, y, t = o.hits_xyt()       # Line # 3 get arrays of x, y, z hit coordinates

        #print 'x:', x
        #print 'y:', y
        #print 't:', t
        #print 'methods:', o.hits_method()

        o.print_hits()               # prints x, y, time, method for found in event hits

    o.print_summary() # print number of events, processing time total, instant and frequency

#------------------------------
#------------------------------    

def test1_HexDataIOExt() :

    # Parameters for initialization of the data source, channels, number of events etc.
    kwargs = {'command'  : 1,
              'srcchs'   : {'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)},
              'numchs'   : 7,
              'numhits'  : 16,
              'dsname'   : 'exp=xpptut15:run=390:smd',
              'evskip'   : 0,
              'events'   : 500,
              'ofprefix' : './',
              'verbose'  : False,
              'cfd_base'        :  0.  ,        
              'cfd_thr'         : -0.04,         
              'cfd_cfr'         :  0.9 ,         
              'cfd_deadtime'    :  5.0 ,    
              'cfd_leadingedge' :  True, 
              'cfd_ioffsetbeg'  :  0   ,  
              'cfd_ioffsetend'  :  1000, 
             }

    ds = psana.MPIDataSource(kwargs['dsname'])
    o = HexDataIOExt(ds, **kwargs)                # Line 1 - object initialization
    o.print_wf_hit_finder_parameters()

    for evt in ds.events() :
        if o.skip_event(evt)           : continue # Line 2 - loop control method passes evt to the object
        if o.event_number() > o.EVENTS : break

        #x, y, t = o.hits_xyt()                   # Line 3 - get arrays x, y, t of hits' coordinates and time
        o.print_hits()                            # prints x, y, time for all hits in the event

    o.print_summary() # print number of events, processing time total, instant and frequency

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    import numpy as np; global np
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print '%s\nTest %s' % (50*'_', tname)
    if   tname=='1' : test1_HexDataIOExt()
    elif tname=='2' : test2_HexDataIOExt()
    else : print 'WARNING: Test %s is not defined' % tname
    sys.exit('End of Test %s' % tname)

#------------------------------
