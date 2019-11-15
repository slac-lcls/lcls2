
#------------------------------

import sys
from copy import deepcopy

import hexanode
import numpy as np
from time import time
from math import sqrt

from psana.hexanode.WFHDF5IO import open_input_h5file
from psana.pyalgos.generic.NDArrUtils import print_ndarr

#------------------------------

def print_tdc_ns(tdc_ns, cmt='  tdc_ns ', fmt=' %7.2f', offset='    ') :
    sh = tdc_ns.shape
    print('%sshape=%s %s' % (offset, str(sh), cmt), end='')
    for r in range(sh[0]) :
        print('\n%sch %1d:' % (offset,r), end='')
        for c in range(min(10,sh[1])) :
             print(fmt % tdc_ns[r,c], end='')
        print
    print('\n%sexit print_tdc_ns\n' % offset)

#------------------------------
if __name__ == "__main__" :

    print(50*'_')

    COMMAND      = 0
    IFNAME       = '/reg/g/psdm/detector/data_test/hdf5/amox27716-r0100-e060000-single-node.h5'
    DETNAME      = 'tmo_hexanode'
    EVSKIP       = 7
    EVENTS       = 10 + EVSKIP
    NUM_CHANNELS = 5
    NUM_HITS     = 6
    CALIBTAB     = '/reg/neh/home4/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/hexanode/examples/calibration_table_data.txt'
    CALIBCFG     = '/reg/neh/home4/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/hexanode/examples/configuration_quad.txt'
    VERBOSE      = True

    #=====================

    file = open_input_h5file(IFNAME)

    #=====================

    print('events in file : %s' % file.h5ds_nevents)
    print('start time     : %s' % file.start_time())
    print('stop time      : %s' % file.stop_time())
    print('tdc_resolution : %s' % file.tdc_resolution())
    print('CALIBTAB       : %s' % CALIBTAB)
    print('CALIBCFG       : %s' % CALIBCFG)

    command = -1;
 
#   // The "command"-value is set in the first line of "sorter.txt"
#   // 0 = only convert to new file format
#   // 1 = sort and write new file 
#   // 2 = calibrate fv, fw, w_offset
#   // 3 = create calibration table files

#   // create the sorter:
    sorter = hexanode.py_sort_class()
    status, command_cfg, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y=\
        hexanode.py_read_config_file(CALIBCFG.encode(), sorter)
    #command = COMMAND # command_cfg
    command = command_cfg

    print('read_config_file status, COMMAND, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y=',\
                            status, command, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y)

    if not status :
        print("WARNING: can't read config file %s" % fname_cfg)
        del sorter
        sys.exit(0)

    print('use_sum_correction', sorter.use_sum_correction)
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

    inds_of_channels    = (Cu1, Cu2, Cv1, Cv2, Cw1, Cw2)
    incr_of_consistence = (  1,   2,   4,   8,  16,  32)
    #inds_of_channels    = (Cu1, Cu2, Cv1, Cv2, Cmcp)
    #incr_of_consistence = (  1,   2,   4,   8,  16)
    inds_incr = list(zip(inds_of_channels, incr_of_consistence))

    #print("chanel increments:", inds_incr)
    
    #=====================

    print("init sorter... ")

    tdc_ns = np.zeros((NUM_CHANNELS, NUM_HITS), dtype=np.float64)
    number_of_hits = np.zeros((NUM_CHANNELS,), dtype=np.int32)

    sorter.set_tdc_resolution_ns(file.tdc_resolution())
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

    #=====================
    # print('ZZZ error_code:', error_code)
    # sys.exit('TEST EXIT')
    #=====================

    if error_code :
        print("sorter could not be initialized\n")
        error_text = sorter.get_error_text(error_code, 512)
        print('Error %d: %s' % (error_code, error_text))
        sys.exit(0)

    print("Calibration factors:\n  f_U (mm/ns) =%f\n  f_V (mm/ns) =%f\n  f_W (mm/ns) =%f\n  Offset on layer W (ns) =%f\n"%\
          (2*sorter.fu, 2*sorter.fv, 2*sorter.fw, w_offset))

    print("ok for sorter initialization\n")

    print("reading event data... \n")

    evnum = 0
    t_sec = time()
    t1_sec = t_sec
    while file.next_event() :
        evnum = file.event_number()

        if evnum < EVSKIP : continue
        if evnum > EVENTS : break

        t1 = time()
        print('Event: %06d, dt(sec): %.3f' % (evnum, t1 - t1_sec))
        t1_sec = t1

#       //==================================
#       // TODO by end user:
#   	// Here you must read in a data block from your data file
#   	// and fill the array tdc_ns[][] and number_of_hits[]

        #nhits = np.zeros((NUMBER_OF_CHANNELS,), dtype=np.int32)
        file.get_number_of_hits_array(number_of_hits, maxvalue=NUM_HITS)
        if file.error_flag() :
            error_text = file.get_error_text(file.error_flag())
            print("file Error %d: %s" % (file.error_flag(), error_text))
            sys.exit(0)

        if VERBOSE : print('====raw number_of_hits_array', number_of_hits[:])
        #number_of_hits = np.array([n if n<NUM_HITS else NUM_HITS for n in number_of_hits])
        #if VERBOSE : print('   number_of_hits_array constrained ', number_of_hits[:8])

        file.get_tdc_data_array(tdc_ns, maxsize=NUM_HITS)

        if file.error_flag() :
            error_text = file.get_error_text(file.error_flag())
            print("file Error %d: %s" % (file.error_flag(), error_text))
            sys.exit(0)

        conds = number_of_hits[:5]==0 
        if conds.any() : continue


        #--------- preserve RAW time sums
        time_sum_u = deepcopy(tdc_ns[Cu1,0] + tdc_ns[Cu2,0] - 2*tdc_ns[Cmcp,0]) #deepcopy(...)
        time_sum_v = deepcopy(tdc_ns[Cv1,0] + tdc_ns[Cv2,0] - 2*tdc_ns[Cmcp,0])
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


        #continue

        print('YYY Point A')
        number_of_particles = sorter.sort() if command == 1 else\
                              sorter.run_without_sorting()
        print('YYY Point B')

        #file.get_tdc_data_array(tdc_ns, NUM_HITS)
        if VERBOSE : print('   sorted number_of_hits_array', number_of_hits[:8])
        if VERBOSE : print_tdc_ns(tdc_ns, cmt='  TDC sorted data ')
        if VERBOSE : print("  Event %5i  number_of_particles: %i" % (evnum, number_of_particles))


        print('XXX number_of_particles:', number_of_particles)


        #=====================
        continue
        #=====================


        if number_of_particles :
          if True :
            for i in range(number_of_particles) :
                hco = hexanode.py_hit_class(sorter, i)
                print("    XXX p:%2i x:%7.3f y:%7.3f t:%7.3f met:%d" % (i, hco.x, hco.y, hco.time, hco.method))
            #print("    part1 u:%7.3f v:%7.3f w:%7.3f" % (u, v, w))


          print('    XXX number_of_particles:', number_of_particles)

        # Discards most of events in command>1
        if number_of_particles<1 : continue





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

        #hco = hexanode.py_hit_class(sorter, 0)

#   	// write the results into a new data file.
#   	// the variable "number_of_particles" contains the number of reconstructed particles.
#   	// the x and y (in mm) and TOF (in ns) is stored in the array sorter->output_hit_array:

#   	// for the i-th particle (i starts from 0):
#       // hco= hexanode.py_hit_class(sorter, i)
#       // hco.x, hco.y, hco.time

#   	// for each particle you can also retrieve the information about how the particle
#   	// was reconstructed (tog et some measure of the confidence):
#   	// hco.method

#   end of the while loop
    print("end of the while loop... \n")

    if command == 2 :
        print("calibrating detector... ")
        sorter.do_calibration()
        print("ok - after do_calibration")

        # QUAD SHOULD NOT USE: scalefactors_calibration_class

        #sfco = hexanode.py_scalefactors_calibration_class(sorter)
        #if sfco :
        #    print("Good calibration factors are:\n  f_U =%f\n  f_V =%f\n  f_W =%f\n  Offset on layer W=%f\n"%\
        #          (2*sorter.fu, 2*sfco.best_fv, 2*sfco.best_fw, sfco.best_w_offset))
        #
        #    print('CALIBRATION: These parameters and time sum offsets from histograms should be set in the file\n  %s' % CALIBCFG)

    if command == 3 : # generate and print(correction tables for sum- and position-correction
        print("creating calibration table in file: %s" % CALIBTAB)
        status = hexanode.py_create_calibration_tables(CALIBTAB, sorter)

        print("CALIBRATION: finished creating calibration tables: %s status %s" % (CALIBTAB, status))


    print("consumed time (sec) = %.6f\n" % (time() - t_sec))

    if sorter is not None : del sorter

    #=====================
    #sys.exit('TEST EXIT')
    #=====================

#------------------------------
