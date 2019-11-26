#!/usr/bin/env python
"""
Module :py:class:`DLDStatistics` for MCP DLD detectors for COLTRIMS experiments
===============================================================================

    from psana.hexanode.DLDStatistics import DLDStatistics

    kwargs = {'STAT_NHITS':True,...}
    proc = DLDProcessor()

    s = DLDStatistics(proc, **kwargs)
    det = ...
    peaks = ...

    for nevt,evt in enumerate(orun.events()):
        wts = det.raw.times(evt)     
        wfs = det.raw.waveforms(evt)
        nhits, pkinds, pkvals, pktns = peaks(wfs,wts) # ACCESS TO PEAK INFO

        s.fill_raw_data(nhits, pktns) 
        s.fill_corrected_data()

Created on 2019-11-20 by Mikhail Dubrovin
"""
#----------
USAGE = 'Run example: python .../psana/hexanode/examples/ex-....py'
#----------

import logging
logger = logging.getLogger(__name__)

import os
import sys
from math import sqrt
import numpy as np

#from psana.pyalgos.generic.NDArrUtils import print_ndarr, info_ndarr
from psana.pyalgos.generic.HBins import HBins
from hexanode import py_hit_class

OSQRT3 = 1./sqrt(3.)
SEC_TO_NS = 1E9

#------------------------------

class DLDStatistics :
    """ holds, fills, and provide access to statistical arrays for MCP DLD data processing
    """
    def __init__(self, proc, **kwargs) :

        self.proc   = proc

        logger.info('In set_parameters, **kwargs: %s' % str(kwargs))
        self.STAT_NHITS         = kwargs.get('STAT_NHITS'        , True)
        self.STAT_TIME_CH       = kwargs.get('STAT_TIME_CH'      , True)
        self.STAT_UVW           = kwargs.get('STAT_UVW'          , True)
        self.STAT_TIME_SUMS     = kwargs.get('STAT_TIME_SUMS'    , True)
        self.STAT_CORRELATIONS  = kwargs.get('STAT_CORRELATIONS' , True)
        self.STAT_XY_COMPONENTS = kwargs.get('STAT_XY_COMPONENTS', True)
        self.STAT_XY_2D         = kwargs.get('STAT_XY_2D'        , True)
        self.STAT_MISC          = kwargs.get('STAT_MISC'         , True)
        self.STAT_REFLECTIONS   = kwargs.get('STAT_REFLECTIONS'  , True)
        self.STAT_PHYSICS       = kwargs.get('STAT_PHYSICS'      , True)
        self.STAT_XY_RESOLUTION = kwargs.get('STAT_XY_RESOLUTION', False) # not available for QUAD

        if self.STAT_TIME_CH :
           self.lst_u1 = []
           self.lst_u2 = []
           self.lst_v1 = []
           self.lst_v2 = []
           self.lst_w1 = []
           self.lst_w2 = []
           self.lst_mcp= []
            
        if self.STAT_NHITS :
           self.lst_nhits_u1 = []
           self.lst_nhits_u2 = []
           self.lst_nhits_v1 = []
           self.lst_nhits_v2 = []
           self.lst_nhits_w1 = []
           self.lst_nhits_w2 = []
           self.lst_nhits_mcp= []
           self.lst_nparts   = []

        if self.STAT_UVW or self.STAT_CORRELATIONS :
           self.lst_u_ns = []
           self.lst_v_ns = []
           self.lst_w_ns = []
           self.lst_u = []
           self.lst_v = []
           self.lst_w = []
        
        if self.STAT_TIME_SUMS or self.STAT_CORRELATIONS :
           self.lst_time_sum_u = []
           self.lst_time_sum_v = []
           self.lst_time_sum_w = []
           
           self.lst_time_sum_u_corr = []
           self.lst_time_sum_v_corr = []
           self.lst_time_sum_w_corr = []

        if self.STAT_XY_COMPONENTS :
           self.lst_Xuv = []
           self.lst_Xuw = []
           self.lst_Xvw = []
           self.lst_Yuv = []
           self.lst_Yuw = []
           self.lst_Yvw = []

        if self.STAT_MISC :
           self.list_dr = []
           self.lst_consist_indicator = []
           self.lst_rec_method = []

        if self.STAT_XY_RESOLUTION :
           self.lst_binx = []
           self.lst_biny = []
           self.lst_resol_fwhm = []

        if self.STAT_REFLECTIONS :
           self.lst_refl_u1 = []
           self.lst_refl_u2 = []
           self.lst_refl_v1 = []
           self.lst_refl_v2 = []
           self.lst_refl_w1 = []
           self.lst_refl_w2 = []

        if self.STAT_XY_2D :
           # images 
           nbins = 360
           self.img_x_bins = HBins((-45., 45.), nbins, vtype=np.float32)
           self.img_y_bins = HBins((-45., 45.), nbins, vtype=np.float32)
           self.img_xy_uv = np.zeros((nbins, nbins), dtype=np.float32)
           self.img_xy_uw = np.zeros((nbins, nbins), dtype=np.float32)
           self.img_xy_vw = np.zeros((nbins, nbins), dtype=np.float32)
           self.img_xy_1  = np.zeros((nbins, nbins), dtype=np.float32)
           self.img_xy_2  = np.zeros((nbins, nbins), dtype=np.float32)

        if self.STAT_PHYSICS :
           t_ns_nbins = 300
           self.t_ns_bins = HBins((1400., 2900.), t_ns_nbins, vtype=np.float32)
           #self.t1_vs_t0 = np.zeros((t_ns_nbins, t_ns_nbins), dtype=np.float32)

           self.ti_vs_tj = np.zeros((t_ns_nbins, t_ns_nbins), dtype=np.float32)
           self.t_all = np.zeros((t_ns_nbins,), dtype=np.float32)
           self.lst_t_all = []

           x_mm_nbins = 200
           y_mm_nbins = 200
           r_mm_nbins = 200
           self.x_mm_bins = HBins((-50., 50.), x_mm_nbins, vtype=np.float32)
           self.y_mm_bins = HBins((-50., 50.), y_mm_nbins, vtype=np.float32)
           self.r_mm_bins = HBins((-50., 50.), r_mm_nbins, vtype=np.float32)
           #self.x_vs_t0 = np.zeros((x_mm_nbins, t_ns_nbins), dtype=np.float32)
           #self.y_vs_t0 = np.zeros((y_mm_nbins, t_ns_nbins), dtype=np.float32)
           self.rsx_vs_t  = np.zeros((r_mm_nbins, t_ns_nbins), dtype=np.float32)
           self.rsy_vs_t  = np.zeros((r_mm_nbins, t_ns_nbins), dtype=np.float32)

#----------

    def fill_data(self, number_of_hits, tdc_sec) :
        self.fill_raw_data(number_of_hits, tdc_sec) 
        self.fill_corrected_data()

#----------

    def fill_raw_data(self, number_of_hits, tdc_sec) :

        tdc_ns = tdc_sec * SEC_TO_NS # sec -> ns

        sorter = self.proc.sorter
        Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp = sorter.channel_indexes

        if self.STAT_NHITS :
           self.lst_nhits_mcp.append(number_of_hits[Cmcp])
           self.lst_nhits_u1. append(number_of_hits[Cu1])
           self.lst_nhits_u2 .append(number_of_hits[Cu2])
           self.lst_nhits_v1 .append(number_of_hits[Cv1])
           self.lst_nhits_v2 .append(number_of_hits[Cv2])
           if sorter.use_hex :        
               self.lst_nhits_w1 .append(number_of_hits[Cw1])
               self.lst_nhits_w2 .append(number_of_hits[Cw2])

        if self.STAT_TIME_CH :
           self.lst_mcp.append(tdc_ns[Cmcp,0])
           self.lst_u1 .append(tdc_ns[Cu1,0])
           self.lst_u2 .append(tdc_ns[Cu2,0])
           self.lst_v1 .append(tdc_ns[Cv1,0])
           self.lst_v2 .append(tdc_ns[Cv2,0])
           if sorter.use_hex :        
               self.lst_w1 .append(tdc_ns[Cw1,0])
               self.lst_w2 .append(tdc_ns[Cw2,0])

        if self.STAT_REFLECTIONS :
           if number_of_hits[Cu2]>1 : self.lst_refl_u1.append(tdc_ns[Cu2,1] - tdc_ns[Cu1,0])
           if number_of_hits[Cu1]>1 : self.lst_refl_u2.append(tdc_ns[Cu1,1] - tdc_ns[Cu2,0])
           if number_of_hits[Cv2]>1 : self.lst_refl_v1.append(tdc_ns[Cv2,1] - tdc_ns[Cv1,0])
           if number_of_hits[Cv1]>1 : self.lst_refl_v2.append(tdc_ns[Cv1,1] - tdc_ns[Cv2,0])
           if sorter.use_hex :        
               if number_of_hits[Cw2]>1 : self.lst_refl_w1.append(tdc_ns[Cw2,1] - tdc_ns[Cw1,0])
               if number_of_hits[Cw1]>1 : self.lst_refl_w2.append(tdc_ns[Cw1,1] - tdc_ns[Cw2,0])

        time_sum_u = tdc_ns[Cu1,0] + tdc_ns[Cu2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_v = tdc_ns[Cv1,0] + tdc_ns[Cv2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_w = tdc_ns[Cw1,0] + tdc_ns[Cw2,0] - 2*tdc_ns[Cmcp,0] if sorter.use_hex else 0

        #logger.info("RAW time_sum_u: %.2f _v: %.2f _w: %.2f " % (time_sum_u, time_sum_v, time_sum_w))

        if self.STAT_TIME_SUMS or self.STAT_CORRELATIONS :
           self.lst_time_sum_u.append(time_sum_u)
           self.lst_time_sum_v.append(time_sum_v)
           self.lst_time_sum_w.append(time_sum_w)

        #if self.STAT_XY_RESOLUTION :
        # NOT VALID FOR QUAD
        #sfco = hexanode.py_scalefactors_calibration_class(sorter) # NOT FOR QUAD
        #    #logger.info("    binx: %d  biny: %d  resolution(FWHM): %.6f" % (sfco.binx, sfco.biny, sfco.detector_map_resol_FWHM_fill))
        #    if sfco.binx>=0 and sfco.biny>=0 :
        #        self.lst_binx.append(sfco.binx)
        #        self.lst_biny.append(sfco.biny)
        #        self.lst_resol_fwhm.append(sfco.detector_map_resol_FWHM_fill)

#----------

    def fill_corrected_data(self) :

        sorter = self.proc.sorter
        Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp = sorter.channel_indexes
        number_of_particles = sorter.output_number_of_hits

        if self.STAT_NHITS :
           self.lst_nparts.append(number_of_particles)

        # Discards most of events in command>1
        #=====================
        if number_of_particles<1 :
            logger.debug('no hits found in event ')
            return
        #=====================

        tdc_ns = self.proc.tdc_ns

        u_ns = tdc_ns[Cu1,0] - tdc_ns[Cu2,0]
        v_ns = tdc_ns[Cv1,0] - tdc_ns[Cv2,0]
        w_ns = 0 #tdc_ns[Cw1,0] - tdc_ns[Cw2,0]

        u = u_ns * sorter.fu
        v = v_ns * sorter.fv
        w = 0 #(w_ns + self.w_offset) * sorter.fw

        Xuv = u
        Xuw = 0 #u
        Xvw = 0 #v + w
        Yuv = v #(u - 2*v)*OSQRT3
        Yuw = 0 #(2*w - u)*OSQRT3
        Yvw = 0 # (w - v)*OSQRT3

        dX = 0 # Xuv - Xvw
        dY = 0 # Yuv - Yvw

        time_sum_u_corr = tdc_ns[Cu1,0] + tdc_ns[Cu2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_v_corr = tdc_ns[Cv1,0] + tdc_ns[Cv2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_w_corr = 0 #tdc_ns[Cw1,0] + tdc_ns[Cw2,0] - 2*tdc_ns[Cmcp,0]

        if sorter.use_hex :        
            w_ns = tdc_ns[Cw1,0] - tdc_ns[Cw2,0]
            w = (w_ns + self.w_offset) * sorter.fw
            
            Xuw = u
            Xvw = v + w
            Yuv = (u - 2*v)*OSQRT3
            Yuw = (2*w - u)*OSQRT3
            Yvw = (w - v)*OSQRT3
            
            dX = Xuv - Xvw
            dY = Yuv - Yvw
            
            time_sum_w_corr = tdc_ns[Cw1,0] + tdc_ns[Cw2,0] - 2*tdc_ns[Cmcp,0]

        #---------

        if self.STAT_UVW or self.STAT_CORRELATIONS :
           self.lst_u_ns.append(u_ns)
           self.lst_v_ns.append(v_ns)
           self.lst_w_ns.append(w_ns)
           self.lst_u.append(u)
           self.lst_v.append(v)
           self.lst_w.append(w)

        if self.STAT_TIME_SUMS or self.STAT_CORRELATIONS :
           self.lst_time_sum_u_corr.append(time_sum_u_corr)
           self.lst_time_sum_v_corr.append(time_sum_v_corr)
           self.lst_time_sum_w_corr.append(time_sum_w_corr)

        if self.STAT_XY_COMPONENTS :
           self.lst_Xuv.append(Xuv)
           self.lst_Xuw.append(Xuw)
           self.lst_Xvw.append(Xvw)
                            
           self.lst_Yuv.append(Yuv)
           self.lst_Yuw.append(Yuw)
           self.lst_Yvw.append(Yvw)

        hco = py_hit_class(sorter, 0)

        if self.STAT_MISC :
           inds_incr = ((Cu1,1), (Cu2,2), (Cv1,4), (Cv2,8), (Cw1,16), (Cw2,32), (Cmcp,64)) if sorter.use_hex else\
                       ((Cu1,1), (Cu2,2), (Cv1,4), (Cv2,8), (Cmcp,16))

           dR = sqrt(dX*dX + dY*dY)
           self.list_dr.append(dR)
           
           # fill Consistence Indicator
           consistenceIndicator = 0
           for (ind, incr) in inds_incr :
             if self.proc.number_of_hits[ind]>0 : consistenceIndicator += incr
           self.lst_consist_indicator.append(consistenceIndicator)

           self.lst_rec_method.append(hco.method)
           #logger.info('reconstruction method %d' % hco.method)


        if self.STAT_XY_2D :

           # fill 2-d images
           x1, y1 = hco.x, hco.y

           x2, y2 = (-10,-10) 
           if number_of_particles > 1 :
               hco2 = py_hit_class(sorter, 1)
               x2, y2 = hco2.x, hco2.y

           ix1, ix2, ixuv, ixuw, ixvw = self.img_x_bins.bin_indexes((x1, x2, Xuv, Xuw, Xvw))
           iy1, iy2, iyuv, iyuw, iyvw = self.img_y_bins.bin_indexes((y1, y2, Yuv, Yuw, Yvw))

           self.img_xy_1 [iy1,  ix1]  += 1
           self.img_xy_2 [iy2,  ix2]  += 1
           self.img_xy_uv[iyuv, ixuv] += 1
           self.img_xy_uw[iyuw, ixuw] += 1 
           self.img_xy_vw[iyvw, ixvw] += 1 

        if self.STAT_PHYSICS :
          if self.proc.number_of_hits[Cmcp]>1 :
            #t0, t1 = tdc_ns[Cmcp,:2]
            #it0, it1 = self.t_ns_bins.bin_indexes((t0, t1))
            #self.t1_vs_t0[it1, it0] += 1

            #ix, iy = self.x_mm_bins.bin_indexes((Xuv,Yuv))
            #self.x_vs_t0[ix, it0] += 1
            #self.y_vs_t0[iy, it0] += 1

            #logger.info("  Event %5i  number_of_particles: %i" % (evnum, number_of_particles))
            #for i in range(number_of_particles) :
            #    hco = py_hit_class(sorter, i) 
            #    #logger.info("    p:%2i x:%7.3f y:%7.3f t:%7.3f met:%d" % (i, hco.x, hco.y, hco.time, hco.method))
            #    x,y,t = hco.x, hco.y, hco.time
            #    r = sqrt(x*x+y*y)
            #    if x<0 : r=-r
            #    ir = self.r_mm_bins.bin_indexes((r,))
            #    it = self.t_ns_bins.bin_indexes((t,))
            #    self.r_vs_t[ir, it] += 1


            for x,y,r,t in sorter.xyrt_list() :
                #irx, iry = self.r_mm_bins.bin_indexes((r if x>0 else -r, r if y>0 else -r))
                iry = self.r_mm_bins.bin_indexes((r if y>0 else -r,))
                it = self.t_ns_bins.bin_indexes((t * SEC_TO_NS,))
                #self.rsx_vs_t[irx, it] += 1
                self.rsy_vs_t[iry, it] += 1

            times = np.array(sorter.t_list()) * SEC_TO_NS
            tinds = self.t_ns_bins.bin_indexes(times) # INDEXES SHOULD BE np.array
            #print_ndarr(times, '\n    XXX times')
            #print_ndarr(tinds, '\n    XXX tinds')

            # accumulate times in the list
            for t in times :
                self.lst_t_all.append(t)

            # accumulate times directly in histogram to evaluate average
            self.t_all[tinds] += 1

            # accumulate times in correlation matrix
            for i in tinds :
                self.ti_vs_tj[i,tinds] += 1

#----------

if __name__ == "__main__" :

    print('%s\n%s'%(50*'_', USAGE))
    #o = DLDStatistics() 

    #=====================
    #sys.exit('TEST EXIT')
    #=====================

#----------

#----------
