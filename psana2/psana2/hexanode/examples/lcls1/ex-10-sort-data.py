#!/usr/bin/env python
#------------------------------
def usage() : return "Use command: python hexanode/examples/ex-10-sort-data.py"
#------------------------------

import os
import sys
import hexanode
import numpy as np
from time import time
from math import sqrt

from pyimgalgos.GlobalUtils import print_ndarr
from expmon.HexDataIO import HexDataIO, do_print
#from pyimgalgos.HBins import HBins

OSQRT3 = 1./sqrt(3.)
#------------------------------

def py_sort(**kwargs) :

    SRCCHS       = kwargs.get('srcchs', {'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)})
    DSNAME       = kwargs.get('dsname', 'exp=xpptut15:run=390:smd') # or h5 file: 'xpptut15-r0390-e300000-n32-mpi.h5'
    EVSKIP       = kwargs.get('evskip', 0)
    EVENTS       = kwargs.get('events', 100) + EVSKIP
    NUM_CHANNELS = kwargs.get('numchs', 7)
    NUM_HITS     = kwargs.get('numhits', 16)
    CALIBTAB     = kwargs.get('calibtab', 'calibration_table_data.txt')
    VERBOSE      = kwargs.get('verbose', False)

    tdc_ns = np.zeros((NUM_CHANNELS, NUM_HITS), dtype=np.float64)
    number_of_hits = np.zeros((NUM_CHANNELS,), dtype=np.int32)

    command = -1;
 
    # The "command"-value is set in the first line of configuration file "sorter_data_cfg.txt"
    # 1 = sort and write new file 
    # 2 = calibrate fv, fw, w_offset
    # 3 = create calibration table files

    # create the sorter object:
    sorter = hexanode.py_sort_class()
    fname_cfg = "sorter_data_cfg.txt"
    status, command, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y=\
        hexanode.py_read_config_file(fname_cfg, sorter)
    print 'read_config_file status, command, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y=',\
                            status, command, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y

    if not status :
        print "WARNING: can't read config file %s" % fname_cfg
        del sorter
        sys.exit(0)

    print 'use_sum_correction', sorter.use_sum_correction
    print 'use_pos_correction', sorter.use_pos_correction
    if sorter is not None :
        if sorter.use_sum_correction or sorter.use_pos_correction :
            status = hexanode.py_read_calibration_tables(CALIBTAB, sorter)

    if command == -1 :
   	print "no config file was read. Nothing to do."
        if sorter is not None : del sorter
        sys.exit(0)

    Cu1  = sorter.cu1 
    Cu2  = sorter.cu2 
    Cv1  = sorter.cv1 
    Cv2  = sorter.cv2 
    Cw1  = sorter.cw1 
    Cw2  = sorter.cw2 
    Cmcp = sorter.cmcp
    print "Numeration of channels - u1:%i  u2:%i  v1:%i  v2:%i  w1:%i  w2:%i  mcp:%i"%\
          (Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp)

    inds_of_channels    = (Cu1, Cu2, Cv1, Cv2, Cw1, Cw2)
    incr_of_consistence = (  1,   2,   4,   8,  16,  32)
    inds_incr = zip(inds_of_channels, incr_of_consistence)
    
    DIO = HexDataIO(srcchs=SRCCHS, numchs=NUM_CHANNELS, numhits=NUM_HITS)

    #=====================
    if '.h5' in DSNAME : DIO.open_input_h5file(DSNAME)
    else :
        DIO.open_input_dataset(DSNAME, pbits=0)

        DIO.set_wf_hit_finder_parameters(**kwargs)
        DIO.print_wf_hit_finder_parameters()
    #=====================

    print 'DIO experiment : %s' % DIO.experiment()
    print 'DIO run        : %s' % DIO.run()
    print 'DIO start time : %s' % DIO.start_time()
    print 'DIO stop time  : %s' % DIO.stop_time()
    print 'DIO tdc_resolution : %.3f' % DIO.tdc_resolution()

    print "init sorter... "

    #sorter.set_tdc_resolution_ns(0.025)
    sorter.set_tdc_resolution_ns(DIO.tdc_resolution())
    sorter.set_tdc_array_row_length(NUM_HITS)
    sorter.set_count(number_of_hits)
    sorter.set_tdc_pointer(tdc_ns)

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
   	print "sorter could not be initialized\n"
        error_text = sorter.get_error_text(error_code, 512)
        print 'Error %d: %s' % (error_code, error_text)
        sys.exit(0)

    print "Calibration factors:\n  f_U (mm/ns) =%f\n  f_V (mm/ns) =%f\n  f_W (mm/ns) =%f\n  Offset on layer W (ns) =%f\n"%\
          (2*sorter.fu, 2*sorter.fv, 2*sorter.fw, w_offset)

    print "ok for sorter initialization\n"

    print "reading event data... \n"

    evnum = 0
    t_sec = time()
    t1_sec = time()
    while DIO.read_next_event() :

        evnum = DIO.event_number()

        if evnum < EVSKIP : continue
        if evnum > EVENTS : break

        if do_print(evnum) :
            t1 = time()
            print 'Event: %06d, dt(sec): %.3f' % (evnum, t1 - t1_sec)
            t1_sec = t1

        #==================================
        # TODO by end user:
        # Here you must read in a data block from your data file
        # and fill the array tdc_ns[][] and number_of_hits[]

        #nhits = np.zeros((NUMBER_OF_CHANNELS,), dtype=np.int32)
        DIO.get_number_of_hits_array(number_of_hits)
        if DIO.error_flag() :
            error_text = DIO.get_error_text(DIO.error_flag())
            print "DIO Error %d: %s" % (DIO.error_flag(), error_text)
            sys.exit(0)
        if VERBOSE : print '   number_of_hits_array', number_of_hits[:8]

        DIO.get_tdc_data_array(tdc_ns)

        if DIO.error_flag() :
            error_text = DIO.get_error_text(DIO.error_flag())
            print "DIO Error %d: %s" % (DIO.error_flag(), error_text)
            sys.exit(0)

        if VERBOSE : print '   TDC data:\n', tdc_ns[0:8,0:5]

        # apply conversion of times to ns
        if False : # DIO returns tdc_ns already in [ns]
            tdc_ns *= DIO.tdc_resolution()

        #==================================
        # NHITS - number of hits per channel
        if True :
            nhits_u1 = number_of_hits[Cu1]
            nhits_u2 = number_of_hits[Cu2]
            nhits_v1 = number_of_hits[Cv1]
            nhits_v2 = number_of_hits[Cv2]
            nhits_w1 = number_of_hits[Cw1]
            nhits_w2 = number_of_hits[Cw2]
            nhits_mcp= number_of_hits[Cmcp]

        # TIME_CH - time of the 1-st hit
        if True :
            t0_u1 = tdc_ns[Cu1,0]
            t0_u2 = tdc_ns[Cu2,0]
            t0_v1 = tdc_ns[Cv1,0]
            t0_v2 = tdc_ns[Cv2,0]
            t0_w1 = tdc_ns[Cw1,0]
            t0_w2 = tdc_ns[Cw2,0]
            t0_mcp= tdc_ns[Cmcp,0]

        # REFLECTIONS
        if True :
            if number_of_hits[Cu2]>1 : refl_u1= tdc_ns[Cu2,1] - tdc_ns[Cu1,0]
            if number_of_hits[Cu1]>1 : refl_u2= tdc_ns[Cu1,1] - tdc_ns[Cu2,0]
            if number_of_hits[Cv2]>1 : refl_v1= tdc_ns[Cv2,1] - tdc_ns[Cv1,0]
            if number_of_hits[Cv1]>1 : refl_v2= tdc_ns[Cv1,1] - tdc_ns[Cv2,0]
            if number_of_hits[Cw2]>1 : refl_w1= tdc_ns[Cw2,1] - tdc_ns[Cw1,0]
            if number_of_hits[Cw1]>1 : refl_w2= tdc_ns[Cw1,1] - tdc_ns[Cw2,0]

        # TIME_SUMS
        time_sum_u = tdc_ns[Cu1,0] + tdc_ns[Cu2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_v = tdc_ns[Cv1,0] + tdc_ns[Cv2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_w = tdc_ns[Cw1,0] + tdc_ns[Cw2,0] - 2*tdc_ns[Cmcp,0]

        # UVW 
        u_ns = tdc_ns[Cu1,0] - tdc_ns[Cu2,0]
        v_ns = tdc_ns[Cv1,0] - tdc_ns[Cv2,0]
        w_ns = tdc_ns[Cw1,0] - tdc_ns[Cw2,0]

        u = u_ns * sorter.fu
        v = v_ns * sorter.fv
        w = (w_ns + w_offset) * sorter.fw

        Xuv = u
        Xuw = u
        Xvw = v + w
        Yuv = (u - 2*v)*OSQRT3
        Yuw = (2*w - u)*OSQRT3
        Yvw = (w - v)*OSQRT3

        dX = Xuv - Xvw
        dY = Yuv - Yvw
        Deviation = sqrt(dX*dX + dY*dY)

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

        #DIO.get_tdc_data_array(tdc_ns)

        time_sum_u_corr = tdc_ns[Cu1,0] + tdc_ns[Cu2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_v_corr = tdc_ns[Cv1,0] + tdc_ns[Cv2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_w_corr = tdc_ns[Cw1,0] + tdc_ns[Cw2,0] - 2*tdc_ns[Cmcp,0]

        #print 'map_is_full_enough', hexanode.py_sorter_scalefactors_calibration_map_is_full_enough(sorter)
        sfco = hexanode.py_scalefactors_calibration_class(sorter)

        # break loop if statistics is enough
        if sfco :
            if sfco.map_is_full_enough() : 
                 print 'sfo.map_is_full_enough(): %s  event number: %06d' % (sfco.map_is_full_enough(), evnum)
                 break

        # XY_RESOLUTION :
        if True :
            #print "    binx: %d  biny: %d  resolution(FWHM): %.6f" % (sfco.binx, sfco.biny, sfco.detector_map_resol_FWHM_fill)
            if sfco.binx>=0 and sfco.biny>=0 :
                binx= sfco.binx
                biny= sfco.biny
                resol_fwhm= sfco.detector_map_resol_FWHM_fill

        # Sort the TDC-Data and reconstruct missing signals and apply the sum- and NL-correction.
        # number_of_particles is the number of reconstructed particles
        #========================================================
   	number_of_particles = sorter.sort() if command == 1 else\
                              sorter.run_without_sorting()
        #========================================================

   	if True :
   	    print "  Event %5i  number_of_particles: %i" % (evnum, number_of_particles)
   	    for i in range(number_of_particles) :
                hco= hexanode.py_hit_class(sorter, i)
   	        print "    p:%1i x:%.3f y:%.3f t:%.3f met:%d" % (i, hco.x, hco.y, hco.time, hco.method)
   	    print "    part1 u:%.3f v:%.3f w:%.3f" % (u, v, w)

        #-------------------------
        # TODO by the end user..."

        if number_of_particles<1 : continue

        hco= hexanode.py_hit_class(sorter, 0)

        # MISC
        if False : 
            # fill Consistence Indicator
            consistenceIndicator = 0
            for (ind, incr) in inds_incr :
              if number_of_hits[ind]>0 : consistenceIndicator += incr
            consist_indicator = consistenceIndicator

            rec_method = hco.method
            #print 'reconstruction method %d' % hco.method

        # XY_2D :
        if False : 
            # fill 2-d images
            x1, y1 = hco.x, hco.y

            x2, y2 = (-10,-10) 
            if number_of_particles > 1 :
                hco2 = hexanode.py_hit_class(sorter, 1)
                x2, y2 = hco2.x, hco2.y

            ix1, ix2, ixuv, ixuw, ixvw = img_x_bins.bin_indexes((x1, x2, Xuv, Xuw, Xvw))
            iy1, iy2, iyuv, iyuw, iyvw = img_y_bins.bin_indexes((y1, y2, Yuv, Yuw, Yvw))

            img_xy_1 [iy1,  ix1]  += 1
            img_xy_2 [iy2,  ix2]  += 1
            img_xy_uv[iyuv, ixuv] += 1
            img_xy_uw[iyuw, ixuw] += 1 
            img_xy_vw[iyvw, ixvw] += 1 

        # PHYSICS :
        if False : 
          if number_of_hits[Cmcp]>1 :
            t0, t1 = tdc_ns[Cmcp,:2]
            it0, it1 = t_ns_bins.bin_indexes((t0, t1))
            t1_vs_t0[it1, it0] += 1

            ix, iy = x_mm_bins.bin_indexes((Xuv,Yuv))
            #iy = y_mm_bins.bin_indexes((Yuv,))
            x_vs_t0[ix, it0] += 1
            y_vs_t0[iy, it0] += 1

#   	// write the results into a new data file.
#   	// the variable "number_of_particles" contains the number of reconstructed particles.
#   	// the x and y (in mm) and TOF (in ns) is stored in the array sorter->output_hit_array:

#   	// for the i-th particle (i starts from 0):
#       // hco= hexanode.py_hit_class(sorter, i)
#       // hco.x, hco.y, hco.time

#   	// for each particle you can also retrieve the information about how the particle
#   	// was reconstructed (tog et some measure of the confidence):
#   	// hco.method

#   end of the event loop

    if command == 2 :
        print "calibrating detector... "
        sorter.do_calibration()
        print "ok - after do_calibration"
        sfco = hexanode.py_scalefactors_calibration_class(sorter)
        if sfco :
            print "Good calibration factors are:\n  f_U =%f\n  f_V =%f\n  f_W =%f\n  Offset on layer W=%f\n"%\
                  (2*sorter.fu, 2*sfco.best_fv, 2*sfco.best_fw, sfco.best_w_offset)

    if command == 3 : # generate and print correction tables for sum- and position-correction
        print "creating calibration tables..."
        status = hexanode.py_create_calibration_tables(CALIBTAB, sorter)
        print "finished creating calibration tables: %s status %s" % (CALIBTAB, status)

    print "consumed time (sec) = %.6f\n" % (time() - t_sec)

    if sorter is not None : del sorter

#------------------------------

if __name__ == "__main__" :
    print 50*'_'
    kwargs = {} # use all default parameters
    py_sort(**kwargs)
    sys.exit('End of %s' % sys.argv[0])

#------------------------------
