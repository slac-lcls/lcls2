#!/usr/bin/env python
"""
Module :py:class:`DLDProcessor` for MCP with DLD for COLTRIMS experiments
=========================================================================

Created on 2019-11-19 by Mikhail Dubrovin
"""
#----------

USAGE = 'Run example: python .../psana/hexanode/examples/ex-16-proc-data.py'

#----------

import logging
logger = logging.getLogger(__name__)

import os
import sys
from time import time
from math import sqrt
import numpy as np

from psana.pyalgos.generic.NDArrUtils import print_ndarr
import psana.pyalgos.generic.Utils as gu
import hexanode

#----------

def print_tdc_ns(tdc_ns, cmt='  tdc_ns ', fmt=' %7.2f', offset='    ') :
    sh = tdc_ns.shape
    print('%sshape=%s %s' % (offset, str(sh), cmt), end='')
    for r in range(sh[0]) :
        print('\n%sch %1d:' % (offset,r), end='')
        for c in range(min(10,sh[1])) :
             print(fmt % tdc_ns[r,c], end='')
        print
    print('\n%sexit print_tdc_ns\n' % offset)

#----------

class DLDProcessor :
    """
    """
    OSQRT3 = 1./sqrt(3.)
    CTYPE_HEX_CONFIG = 'hex_config'
    CTYPE_HEX_TABLE  = 'hex_table'
        
    def __init__(self, **kwargs) :
        print('__init__, **kwargs: %s' % str(kwargs))
        #print(gu.str_kwargs(kwargs, title='input parameters:'))

        #DSNAME       = kwargs.get('dsname', '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2')
        #COMMAND      = kwargs.get('command', 0)
        #DETNAME      = kwargs.get('detname', 'tmo_hexanode')
        #EVSKIP       = kwargs.get('evskip', 0)
        #EVENTS       = kwargs.get('events', 1000000) + EVSKIP
        NUM_CHANNELS      = kwargs.get('numchs', 5)
        NUM_HITS          = kwargs.get('numhits', 16)
        OFPREFIX          = kwargs.get('ofprefix','./figs-hexanode/plot')
        
        self.VERBOSE      = kwargs.get('verbose', False)
        calibtab          = kwargs.get('calibtab', None)
        calibcfg          = kwargs.get('calibcfg', None)
        CALIBCFG          = calibcfg #if calibcfg is not None else file.find_calib_file(type=self.CTYPE_HEX_CONFIG)
        CALIBTAB          = calibtab #if calibtab is not None else file.find_calib_file(type=self.CTYPE_HEX_TABLE)

        TDC_RESOLUTION = kwargs.get('tdc_resolution', 0.250) # ns !!! SHOULD BE TAKEN FROM DETECTOR CONFIGURATION 

#------------------------------

        #create_output_directory(OFPREFIX)

        print('TDC_RESOLUTION : %s' % TDC_RESOLUTION)
        print('CALIBTAB       : %s' % CALIBTAB)
        print('CALIBCFG       : %s' % CALIBCFG)

        #print('file calib_dir   : %s' % file.calib_dir())
        #print('file calib_src   : %s' % file.calib_src())
        #print('file calib_group : %s' % file.calib_group())
        #print('file ctype_dir   : %s' % file.calibtype_dir())


#       // The "command"-value is set in the first line of "sorter.txt"
#       // 0 = only convert to new file format
#       // 1 = sort and write new file 
#       // 2 = calibrate fv, fw, w_offset
#       // 3 = create calibration table files

#   // create the sorter:
        sorter = self.sorter = hexanode.py_sort_class()
        status, command_cfg, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y=\
            hexanode.py_read_config_file(CALIBCFG.encode(), sorter)

        self.command = command = command_cfg

        print('read_config_file status, COMMAND, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y=',\
                                status, command, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y)

        if not status :
            print("WARNING: can't read config file %s" % CALIBCFG)
            del sorter
            sys.exit(0)

        print('use_sum_correction',          sorter.use_sum_correction)
        print('use_pos_correction HEX ONLY', sorter.use_pos_correction)
        if sorter is not None :
            if sorter.use_sum_correction or sorter.use_pos_correction :
                status = hexanode.py_read_calibration_tables(CALIBTAB.encode(), sorter)

        if command == -1 :
            print("no config file was read. Nothing to do.")
            if sorter is not None : del sorter
            sys.exit(0)

        Cu1  = sorter.cu1 
        Cu2  = sorter.cu2 
        Cv1  = sorter.cv1 
        Cv2  = sorter.cv2 
        Cw1  = sorter.cw1 
        Cw2  = sorter.cw2 
        Cmcp = sorter.cmcp
        print("Numeration of channels - u1:%i  u2:%i  v1:%i  v2:%i  w1:%i  w2:%i  mcp:%i"%\
              (Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp))

        self.inds_incr = ((Cu1,1), (Cu2,2), (Cv1,4), (Cv2,8), (Cw1,16), (Cw2,32), (Cmcp,64)) if sorter.use_hex else\
                         ((Cu1,1), (Cu2,2), (Cv1,4), (Cv2,8), (Cmcp,16))
        #print("chanel increments:", self.inds_incr)
    
        #=====================

        print("init sorter... ")

        self.tdc_ns = np.zeros((NUM_CHANNELS, NUM_HITS), dtype=np.float64)
        self.number_of_hits = np.zeros((NUM_CHANNELS,), dtype=np.int32)

        sorter.set_tdc_resolution_ns(TDC_RESOLUTION)
        sorter.set_tdc_array_row_length(NUM_HITS)
        sorter.set_count(self.number_of_hits)
        sorter.set_tdc_pointer(self.tdc_ns)

        #sorter.set_use_reflection_filter_on_u1(False) # Achim recommended False
        #sorter.set_use_reflection_filter_on_u2(False)

        self.on_command_23_init()

        error_code = sorter.init_after_setting_parameters()
        if error_code :
            print("sorter could not be initialized\n")
            error_text = sorter.get_error_text(error_code, 512)
            print('Error %d: %s' % (error_code, error_text))
            sys.exit(0)

        print("Calibration factors:\n  f_U (mm/ns) =%f\n  f_V (mm/ns) =%f\n  f_W (mm/ns) =%f\n  Offset on layer W (ns) =%f\n"%\
              (2*sorter.fu, 2*sorter.fv, 2*sorter.fw, w_offset))

        print("ok for sorter initialization\n")

        self.evnum_old = None








    def set_data_arrays(self, nhits, pktns) :
        NUM_CHANNELS, NUM_HITS = self.tdc_ns.shape
        conds = nhits[:NUM_CHANNELS]==0 
        if conds.any() :
            logger.warning('array number_of_hits has channels with zero hits: %s'%str(nhits))
            return False

        self.number_of_hits[:NUM_CHANNELS] = nhits[:NUM_CHANNELS]
        for c in range(NUM_CHANNELS) :
            self.tdc_ns[c,:nhits[c]] = pktns[c,:nhits[c]]

        logger.debug('XXX number_of_hits: %s'%str(nhits))
        return True





    def event_proc(self, evnum, nhits, pktns) :
        logger.debug('XXX A event_proc %s'%str(nhits))

        if evnum == self.evnum_old : return
        self.evnum_old = evnum

        logger.debug('XXX B event_proc %s'%str(nhits))

#       //==================================
#       // TODO by end user:
#   	// Here you must read in a data block from your data file
#   	// and fill the array tdc_ns[][] and number_of_hits[]

        if not self.set_data_arrays(nhits, pktns) : return

        #==========
        return
        #==========

        if sp.PLOT_NHITS :
            sp.lst_nhits_u1. append(number_of_hits[Cu1])
            sp.lst_nhits_u2 .append(number_of_hits[Cu2])
            sp.lst_nhits_v1 .append(number_of_hits[Cv1])
            sp.lst_nhits_v2 .append(number_of_hits[Cv2])
            #sp.lst_nhits_w1 .append(number_of_hits[Cw1])
            #sp.lst_nhits_w2 .append(number_of_hits[Cw2])
            sp.lst_nhits_mcp.append(number_of_hits[Cmcp])


        if sp.PLOT_TIME_CH :
            sp.lst_u1 .append(tdc_ns[Cu1,0])
            sp.lst_u2 .append(tdc_ns[Cu2,0])
            sp.lst_v1 .append(tdc_ns[Cv1,0])
            sp.lst_v2 .append(tdc_ns[Cv2,0])
            #sp.lst_w1 .append(tdc_ns[Cw1,0])
            #sp.lst_w2 .append(tdc_ns[Cw2,0])
            sp.lst_mcp.append(tdc_ns[Cmcp,0])

        if sp.PLOT_REFLECTIONS :
            if number_of_hits[Cu2]>1 : sp.lst_refl_u1.append(tdc_ns[Cu2,1] - tdc_ns[Cu1,0])
            if number_of_hits[Cu1]>1 : sp.lst_refl_u2.append(tdc_ns[Cu1,1] - tdc_ns[Cu2,0])
            if number_of_hits[Cv2]>1 : sp.lst_refl_v1.append(tdc_ns[Cv2,1] - tdc_ns[Cv1,0])
            if number_of_hits[Cv1]>1 : sp.lst_refl_v2.append(tdc_ns[Cv1,1] - tdc_ns[Cv2,0])
            #if number_of_hits[Cw2]>1 : sp.lst_refl_w1.append(tdc_ns[Cw2,1] - tdc_ns[Cw1,0])
            #if number_of_hits[Cw1]>1 : sp.lst_refl_w2.append(tdc_ns[Cw1,1] - tdc_ns[Cw2,0])


        #--------- preserve RAW time sums
        #time_sum_u = deepcopy(tdc_ns[Cu1,0] + tdc_ns[Cu2,0] - 2*tdc_ns[Cmcp,0]) #deepcopy(...)
        #time_sum_v = deepcopy(tdc_ns[Cv1,0] + tdc_ns[Cv2,0] - 2*tdc_ns[Cmcp,0])
        time_sum_u = tdc_ns[Cu1,0] + tdc_ns[Cu2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_v = tdc_ns[Cv1,0] + tdc_ns[Cv2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_w = 0 #tdc_ns[Cw1,0] + tdc_ns[Cw2,0] - 2*tdc_ns[Cmcp,0]

        #print("RAW time_sum_u, time_sum_v:", time_sum_u, time_sum_v)
        #---------

        if VERBOSE : print_tdc_ns(tdc_ns, cmt='  TDC raw data ')

        if sorter.use_hex :        
  	    # shift the time sums to zero:
            sorter.shift_sums(+1, offset_sum_u, offset_sum_v, offset_sum_w)
   	    #shift layer w so that the middle lines of all layers intersect in one point:
            sorter.shift_layer_w(+1, w_offset)
        else :
            # shift the time sums to zero:
            sorter.shift_sums(+1, offset_sum_u, offset_sum_v)

        if VERBOSE : print_tdc_ns(tdc_ns, cmt='  TDC after shift_sums ')

        # shift all signals from the anode so that the center of the detector is at x=y=0:
        sorter.shift_position_origin(+1, pos_offset_x, pos_offset_y)
        sorter.feed_calibration_data(True, w_offset) # for calibration of fv, fw, w_offset and correction tables

        if VERBOSE : print_tdc_ns(tdc_ns, cmt='  TDC after feed_calibration_data ')

        #print('map_is_full_enough', hexanode.py_sorter_scalefactors_calibration_map_is_full_enough(sorter))

        # NOT VALID FOR QUAD
        #sfco = hexanode.py_scalefactors_calibration_class(sorter) # NOT FOR QUAD
        # break loop if statistics is enough
        #if sfco :
        #    if sfco.map_is_full_enough() : 
        #         print('sfo.map_is_full_enough(): %s  event number: %06d' % (sfco.map_is_full_enough(), evnum))
        #         break

        #if sp.PLOT_XY_RESOLUTION :
        #    #print("    binx: %d  biny: %d  resolution(FWHM): %.6f" % (sfco.binx, sfco.biny, sfco.detector_map_resol_FWHM_fill))
        #    if sfco.binx>=0 and sfco.biny>=0 :
        #        sp.lst_binx.append(sfco.binx)
        #        sp.lst_biny.append(sfco.biny)
        #        sp.lst_resol_fwhm.append(sfco.detector_map_resol_FWHM_fill)

        # Sort the TDC-Data and reconstruct missing signals and apply the time-sum- and NL-correction.
        # number_of_particles is the number of reconstructed particles

        number_of_particles = sorter.sort() if command == 1 else\
                              sorter.run_without_sorting()

        #file.get_tdc_data_array(tdc_ns, NUM_HITS)
        if VERBOSE : print('    (un/)sorted number_of_hits_array', number_of_hits[:8])
        if VERBOSE : print_tdc_ns(tdc_ns, cmt='  TDC sorted data ')
        if VERBOSE : 
            print("  Event %5i  number_of_particles: %i" % (evnum, number_of_particles))
            for i in range(number_of_particles) :
                #### IT DID NOT WORK ON LCLS2 because pointer was deleted in py_hit_class.__dealloc__
                hco = hexanode.py_hit_class(sorter, i) 
                print("    p:%2i x:%7.3f y:%7.3f t:%7.3f met:%d" % (i, hco.x, hco.y, hco.time, hco.method))
 
        #print_tdc_ns(tdc_ns, cmt='  TDC sorted data ')
        #print('    XXX sorter.time_list', sorter.t_list())

        if sp.PLOT_NHITS :
            sp.lst_nparts.append(number_of_particles)

        # Discards most of events in command>1
        #=====================
        if number_of_particles<1 :
            return
        #=====================

#       // TODO by end user..."

        #---------

        u_ns = tdc_ns[Cu1,0] - tdc_ns[Cu2,0]
        v_ns = tdc_ns[Cv1,0] - tdc_ns[Cv2,0]
        w_ns = 0 #tdc_ns[Cw1,0] - tdc_ns[Cw2,0]

        u = u_ns * sorter.fu
        v = v_ns * sorter.fv
        w = 0 #(w_ns + w_offset) * sorter.fw

        Xuv = u
        Xuw = 0 #u
        Xvw = 0 #v + w
        Yuv = v #(u - 2*v)*OSQRT3
        Yuw = 0 #(2*w - u)*OSQRT3
        Yvw = 0 # (w - v)*OSQRT3

        dX = 0 # Xuv - Xvw
        dY = 0 # Yuv - Yvw
        dR = sqrt(dX*dX + dY*dY)

        time_sum_u_corr = tdc_ns[Cu1,0] + tdc_ns[Cu2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_v_corr = tdc_ns[Cv1,0] + tdc_ns[Cv2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_w_corr = 0 #tdc_ns[Cw1,0] + tdc_ns[Cw2,0] - 2*tdc_ns[Cmcp,0]

        #---------

        if sp.PLOT_UVW or sp.PLOT_CORRELATIONS :
            sp.lst_u_ns.append(u_ns)
            sp.lst_v_ns.append(v_ns)
            sp.lst_w_ns.append(w_ns)
            sp.lst_u.append(u)
            sp.lst_v.append(v)
            sp.lst_w.append(w)

        if sp.PLOT_TIME_SUMS or sp.PLOT_CORRELATIONS :
            sp.lst_time_sum_u.append(time_sum_u)
            sp.lst_time_sum_v.append(time_sum_v)
            sp.lst_time_sum_w.append(time_sum_w)

            sp.lst_time_sum_u_corr.append(time_sum_u_corr)
            sp.lst_time_sum_v_corr.append(time_sum_v_corr)
            sp.lst_time_sum_w_corr.append(time_sum_w_corr)

        if sp.PLOT_XY_COMPONENTS :
            sp.lst_Xuv.append(Xuv)
            sp.lst_Xuw.append(Xuw)
            sp.lst_Xvw.append(Xvw)
                             
            sp.lst_Yuv.append(Yuv)
            sp.lst_Yuw.append(Yuw)
            sp.lst_Yvw.append(Yvw)


        hco = hexanode.py_hit_class(sorter, 0)

        if sp.PLOT_MISC :
            sp.list_dr.append(dR)
            
            # fill Consistence Indicator
            consistenceIndicator = 0
            for (ind, incr) in self.inds_incr :
              if number_of_hits[ind]>0 : consistenceIndicator += incr
            sp.lst_consist_indicator.append(consistenceIndicator)

            sp.lst_rec_method.append(hco.method)
            #print('reconstruction method %d' % hco.method)


        if sp.PLOT_XY_2D :

            # fill 2-d images
            x1, y1 = hco.x, hco.y

            x2, y2 = (-10,-10) 
            if number_of_particles > 1 :
                hco2 = hexanode.py_hit_class(sorter, 1)
                x2, y2 = hco2.x, hco2.y

            ix1, ix2, ixuv, ixuw, ixvw = sp.img_x_bins.bin_indexes((x1, x2, Xuv, Xuw, Xvw))
            iy1, iy2, iyuv, iyuw, iyvw = sp.img_y_bins.bin_indexes((y1, y2, Yuv, Yuw, Yvw))

            sp.img_xy_1 [iy1,  ix1]  += 1
            sp.img_xy_2 [iy2,  ix2]  += 1
            sp.img_xy_uv[iyuv, ixuv] += 1
            sp.img_xy_uw[iyuw, ixuw] += 1 
            sp.img_xy_vw[iyvw, ixvw] += 1 

        if sp.PLOT_PHYSICS :
          if number_of_hits[Cmcp]>1 :
            #t0, t1 = tdc_ns[Cmcp,:2]
            #it0, it1 = sp.t_ns_bins.bin_indexes((t0, t1))
            #sp.t1_vs_t0[it1, it0] += 1

            #ix, iy = sp.x_mm_bins.bin_indexes((Xuv,Yuv))
            #sp.x_vs_t0[ix, it0] += 1
            #sp.y_vs_t0[iy, it0] += 1

            #print("  Event %5i  number_of_particles: %i" % (evnum, number_of_particles))
            #for i in range(number_of_particles) :
            #    hco = hexanode.py_hit_class(sorter, i) 
            #    #print("    p:%2i x:%7.3f y:%7.3f t:%7.3f met:%d" % (i, hco.x, hco.y, hco.time, hco.method))
            #    x,y,t = hco.x, hco.y, hco.time
            #    r = sqrt(x*x+y*y)
            #    if x<0 : r=-r
            #    ir = sp.r_mm_bins.bin_indexes((r,))
            #    it = sp.t_ns_bins.bin_indexes((t,))
            #    sp.r_vs_t[ir, it] += 1


            for x,y,r,t in sorter.xyrt_list() :
                irx, iry = sp.r_mm_bins.bin_indexes((r if x>0 else -r, r if y>0 else -r))
                it = sp.t_ns_bins.bin_indexes((t,))
                sp.rsx_vs_t[irx, it] += 1
                sp.rsy_vs_t[iry, it] += 1

            times = sorter.t_list()
            tinds = sp.t_ns_bins.bin_indexes(times) # INDEXES SHOULD BE np.array
            #print_ndarr(times, '\n    XXX times')
            #print_ndarr(tinds, '\n    XXX tinds')

            # accumulate times in the list
            for t in times :
                sp.lst_t_all.append(t)

            # accumulate times directly in histogram to evaluate average
            sp.t_all[tinds] += 1

            # accumulate times in correlation matrix
            for i in tinds :
                sp.ti_vs_tj[i,tinds] += 1




#   end of the while loop
        print("end of the while loop... \n")
        
        self.on_command_2_end()
        self.on_command_3_end()
        
        
    def on_command_23_init(self) :
        if self.command >= 2 :
            sorter = self.sorter
            sorter.create_scalefactors_calibrator(True,\
                                                  sorter.runtime_u,\
                                                  sorter.runtime_v,\
                                                  sorter.runtime_w, 0.78,\
                                                  sorter.fu, sorter.fv, sorter.fw)

    def on_command_2_end(self) :
      if self.command == 2 :
        print("sorter.do_calibration()... for command=2")
        self.sorter.do_calibration()
        print("ok - after do_calibration")

        # QUAD SHOULD NOT USE: scalefactors_calibration_class

        #sfco = hexanode.py_scalefactors_calibration_class(sorter)
        #if sfco :
        #    print("Good calibration factors are:\n  f_U =%f\n  f_V =%f\n  f_W =%f\n  Offset on layer W=%f\n"%\
        #          (2*sorter.fu, 2*sfco.best_fv, 2*sfco.best_fw, sfco.best_w_offset))
        #
        #    print('CALIBRATION: These parameters and time sum offsets from histograms should be set in the file\n  %s' % CALIBCFG)


    def on_command_3_end(self) :
      if self.command == 3 : # generate and print(correction tables for sum- and position-correction
        CALIBTAB = calibtab if calibtab is not None else\
                   file.make_calib_file_path(type=CTYPE_HEX_TABLE)
        print("creating calibration table in file: %s" % CALIBTAB)
        status = hexanode.py_create_calibration_tables(CALIBTAB.encode(), sorter)
        print("CALIBRATION: finished creating calibration tables: status %s" % status)


    def __del__(self) :
        if self.sorter is not None : del self.sorter

#----------

if __name__ == "__main__" :
    print(50*'_')
    print(USAGE)

    #kwargs = {'events':1500,}
    #calib_on_data(**kwargs)

    #=====================
    #sys.exit('TEST EXIT')
    #=====================

#----------
