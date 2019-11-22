#!/usr/bin/env python
"""
Module :py:class:`DLDProcessor` for MCP with DLD for COLTRIMS experiments
=========================================================================

    from psana.hexanode.DLDProcessor import DLDProcessor

    kwargs = {'events':1500,...}
    peaks = WFPeaks(**kwargs)
    o  = DLDProcessor(**kwargs)

    ds    = DataSource(files=DSNAME)
    orun  = next(ds.runs())
    det   = orun.Detector(DETNAME)

    for nevt,evt in enumerate(orun.events()):
        wts = det.raw.times(evt)     
        wfs = det.raw.waveforms(evt)
        nhits, pkinds, pkvals, pktns = peaks(wfs,wts) # ACCESS TO PEAK INFO
        xyrt = o.xyrt_list(nevt, nhits, pktns)

Created on 2019-11-20 by Mikhail Dubrovin
"""
#----------

USAGE = 'Run example: python .../psana/hexanode/examples/ex-16-proc-data.py'

#----------

import logging
logger = logging.getLogger(__name__)

import sys
from time import time
from math import sqrt
import numpy as np

from psana.pyalgos.generic.NDArrUtils import print_ndarr, info_ndarr
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
        logger.info('__init__, **kwargs: %s' % str(kwargs))
        #logger.info(gu.str_kwargs(kwargs, title='input parameters:'))

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

        logger.info('TDC_RESOLUTION : %s' % TDC_RESOLUTION)
        logger.info('CALIBTAB       : %s' % CALIBTAB)
        logger.info('CALIBCFG       : %s' % CALIBCFG)

        #logger.info('file calib_dir   : %s' % file.calib_dir())
        #logger.info('file calib_src   : %s' % file.calib_src())
        #logger.info('file calib_group : %s' % file.calib_group())
        #logger.info('file ctype_dir   : %s' % file.calibtype_dir())


#       // The "command"-value is set in the first line of "sorter.txt"
#       // 0 = only convert to new file format
#       // 1 = sort and write new file 
#       // 2 = calibrate fv, fw, w_offset
#       // 3 = create calibration table files

#   // create the sorter:
        self.sorter = sorter = hexanode.py_sort_class()
        status, command_cfg, self.offset_sum_u, self.offset_sum_v, self.offset_sum_w, self.w_offset, self.pos_offset_x, self.pos_offset_y=\
            hexanode.py_read_config_file(CALIBCFG.encode(), sorter)

        self.command = command = command_cfg
        
        logger.info('read_config_file status:%s COMMAND:%d offset_sum_u:%.3f offset_sum_v:%.3f offset_sum_w:%.3f w_offset:%.3f pos_offset_x:%.3f pos_offset_y:%.3f',\
                    status, command, self.offset_sum_u, self.offset_sum_v, self.offset_sum_w, self.w_offset, self.pos_offset_x, self.pos_offset_y)

        if not status :
            logger.info("WARNING: can't read config file %s" % CALIBCFG)
            del sorter
            sys.exit(0)

        logger.info('use_sum_correction          %s' % sorter.use_sum_correction)
        logger.info('use_pos_correction HEX ONLY %s' % sorter.use_pos_correction)
        if sorter is not None :
            if sorter.use_sum_correction or sorter.use_pos_correction :
                status = hexanode.py_read_calibration_tables(CALIBTAB.encode(), sorter)

        if command == -1 :
            logger.info("no config file was read. Nothing to do.")
            if sorter is not None : del sorter
            sys.exit(0)

        Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp = sorter.channel_indexes
        logger.info("Numeration of channels - u1:%i  u2:%i  v1:%i  v2:%i  w1:%i  w2:%i  mcp:%i"%\
              (Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp))

        self.inds_incr = ((Cu1,1), (Cu2,2), (Cv1,4), (Cv2,8), (Cw1,16), (Cw2,32), (Cmcp,64)) if sorter.use_hex else\
                         ((Cu1,1), (Cu2,2), (Cv1,4), (Cv2,8), (Cmcp,16))

        #logger.info("chanel increments:", self.inds_incr)
    
        #=====================

        logger.info("init sorter... ")

        self.tdc_ns = np.zeros((NUM_CHANNELS, NUM_HITS), dtype=np.float64)
        self.number_of_hits = np.zeros((NUM_CHANNELS,), dtype=np.int32)

        sorter.set_tdc_resolution_ns(TDC_RESOLUTION)
        sorter.set_tdc_array_row_length(NUM_HITS)
        sorter.set_count(self.number_of_hits)
        sorter.set_tdc_pointer(self.tdc_ns)

        #sorter.set_use_reflection_filter_on_u1(False) # Achim recommended False
        #sorter.set_use_reflection_filter_on_u2(False)

        self._on_command_23_init()

        error_code = sorter.init_after_setting_parameters()
        if error_code :
            logger.info("sorter could not be initialized\n")
            error_text = sorter.get_error_text(error_code, 512)
            logger.info('Error %d: %s' % (error_code, error_text))
            sys.exit(0)

        logger.info("Calibration factors:\n  f_U (mm/ns) =%f\n  f_V (mm/ns) =%f\n  f_W (mm/ns) =%f\n  Offset on layer W (ns) =%f\n"%\
              (2*sorter.fu, 2*sorter.fv, 2*sorter.fw, self.w_offset))

        logger.info("ok for sorter initialization\n")

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

        logger.info(info_ndarr(self.number_of_hits, 'number_of_hits'))
        logger.info(info_ndarr(self.tdc_ns,         'tdc_ns'))

        return True


    def event_proc(self, evnum, nhits, pktns) :
        """
           TODO by end user:
           Here you must read in a data block from your data file
           and fill the array tdc_ns[][] and number_of_hits[]
        """
        if evnum == self.evnum_old : return
        self.evnum_old = evnum

        if not self.set_data_arrays(nhits, pktns) : return

        sorter, number_of_hits, tdc_ns  = self.sorter, self.number_of_hits, self.tdc_ns

        Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp = sorter.channel_indexes

        if self.VERBOSE : print_tdc_ns(tdc_ns, cmt='  TDC raw data ')

        if sorter.use_hex :        
  	    # shift the time sums to zero:
            sorter.shift_sums(+1, self.offset_sum_u, self.offset_sum_v, self.offset_sum_w)
   	    #shift layer w so that the middle lines of all layers intersect in one point:
            sorter.shift_layer_w(+1, self.w_offset)
        else :
            # shift the time sums to zero:
            sorter.shift_sums(+1, self.offset_sum_u, self.offset_sum_v)

        if self.VERBOSE : print_tdc_ns(tdc_ns, cmt='  TDC after shift_sums ')

        # shift all signals from the anode so that the center of the detector is at x=y=0:
        sorter.shift_position_origin(+1, self.pos_offset_x, self.pos_offset_y)
        sorter.feed_calibration_data(True, self.w_offset) # for calibration of fv, fw, w_offset and correction tables

        if self.VERBOSE : print_tdc_ns(tdc_ns, cmt='  TDC after feed_calibration_data ')

        #logger.info('map_is_full_enough', hexanode.py_sorter_scalefactors_calibration_map_is_full_enough(sorter))

        # NOT VALID FOR QUAD
        #sfco = hexanode.py_scalefactors_calibration_class(sorter) # NOT FOR QUAD
        # break loop if statistics is enough
        #if sfco :
        #    if sfco.map_is_full_enough() : 
        #         logger.info('sfo.map_is_full_enough(): %s  event number: %06d' % (sfco.map_is_full_enough(), evnum))
        #         break

        # Sort the TDC-Data and reconstruct missing signals and apply the time-sum- and NL-correction.
        # number_of_particles is the number of reconstructed particles

        number_of_particles = sorter.sort() if self.command == 1 else\
                              sorter.run_without_sorting()

        #file.get_tdc_data_array(tdc_ns, NUM_HITS)
        if self.VERBOSE :
            logger.info('  (un/)sorted number_of_hits_array %s' % str(number_of_hits[:8]))
            print_tdc_ns(tdc_ns, cmt='  TDC sorted data ')
            logger.info("  Event %5i  number_of_particles: %i" % (evnum, number_of_particles))
            for i in range(number_of_particles) :
                #### IT DID NOT WORK ON LCLS2 because pointer was deleted in py_hit_class.__dealloc__
                hco = hexanode.py_hit_class(sorter, i) 
                logger.info("    p:%2i x:%7.3f y:%7.3f t:%7.3f met:%d" % (i, hco.x, hco.y, hco.time, hco.method))
 
        #print_tdc_ns(tdc_ns, cmt='  TDC sorted data ')
        #logger.info('    XXX sorter.time_list', sorter.t_list())


    def xyrt_list(self, evnum, nhits, pktns) :
        if evnum != self.evnum_old : self.event_proc(evnum, nhits, pktns)
        return self.sorter.xyrt_list()


    def xyt_list(self, evnum, nhits, pktns) :
        if evnum != self.evnum_old : self.event_proc(evnum, nhits, pktns)
        return self.sorter.xyt_list()


    def _on_command_23_init(self) :
        if self.command >= 2 :
            sorter = self.sorter
            sorter.create_scalefactors_calibrator(True,\
                                                  sorter.runtime_u,\
                                                  sorter.runtime_v,\
                                                  sorter.runtime_w, 0.78,\
                                                  sorter.fu, sorter.fv, sorter.fw)

    def _on_command_2_end(self) :
      if self.command == 2 :
        logger.info("sorter.do_calibration()... for command=2")
        self.sorter.do_calibration()
        logger.info("ok - after do_calibration")

        # QUAD SHOULD NOT USE: scalefactors_calibration_class
        #sfco = hexanode.py_scalefactors_calibration_class(sorter)
        #if sfco :
        #    logger.info("Good calibration factors are:\n  f_U =%f\n  f_V =%f\n  f_W =%f\n  Offset on layer W=%f\n"%\
        #          (2*sorter.fu, 2*sfco.best_fv, 2*sfco.best_fw, sfco.best_w_offset))
        #
        #    logger.info('CALIBRATION: These parameters and time sum offsets from histograms should be set in the file\n  %s' % CALIBCFG)


    def _on_command_3_end(self) :
      if self.command == 3 : # generate and logger.info(correction tables for sum- and position-correction
        CALIBTAB = calibtab if calibtab is not None else\
                   file.make_calib_file_path(type=CTYPE_HEX_TABLE)
        logger.info("creating calibration table in file: %s" % CALIBTAB)
        status = hexanode.py_create_calibration_tables(CALIBTAB.encode(), sorter)
        logger.info("CALIBRATION: finished creating calibration tables: status %s" % status)


    def end_proc(self) :
        logger.info('end_proc - end of the event loop... \n')  
        self._on_command_2_end()
        self._on_command_3_end()
        
        
    def __del__(self) :
        self.end_proc()
        if self.sorter is not None : del self.sorter

#----------

if __name__ == "__main__" :
    print('%s\n%s'%(50*'_', USAGE))

    #=====================
    #sys.exit('TEST EXIT')
    #=====================

#----------
