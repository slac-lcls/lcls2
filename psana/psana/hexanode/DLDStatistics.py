#!/usr/bin/env python
"""
Module :py:class:`DLDStatistics` for MCP DLD detectors for COLTRIMS experiments
===============================================================================

    from psana.hexanode.DLDStatistics import DLDStatistics

    kwargs = {'STAT_NHITS':True,...}
    sorter = ...

    o = DLDStatistics(sorter, **kwargs)
    det = ...
    peaks = ...

    for nevt,evt in enumerate(orun.events()):
        wts = det.raw.times(evt)     
        wfs = det.raw.waveforms(evt)
        nhits, pkinds, pkvals, pktns = peaks(wfs,wts) # ACCESS TO PEAK INFO

        o.fill_raw_data(nhits, pktns) 
        o.fill_corrected_data()

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

#------------------------------

class DLDStatistics :
    """ holds, fills, and provide access to statistical arrays for MCP DLD data processing
    """
    def __init__(sp, sorter, **kwargs) :

        sp.sorter = sorter

        logger.info('In set_parameters, **kwargs: %s' % str(kwargs))
        sp.STAT_NHITS         = kwargs.get('STAT_NHITS'        , True)
        sp.STAT_TIME_CH       = kwargs.get('STAT_TIME_CH'      , True)
        sp.STAT_UVW           = kwargs.get('STAT_UVW'          , True)
        sp.STAT_TIME_SUMS     = kwargs.get('STAT_TIME_SUMS'    , True)
        sp.STAT_CORRELATIONS  = kwargs.get('STAT_CORRELATIONS' , True)
        sp.STAT_XY_COMPONENTS = kwargs.get('STAT_XY_COMPONENTS', True)
        sp.STAT_XY_2D         = kwargs.get('STAT_XY_2D'        , True)
        sp.STAT_MISC          = kwargs.get('STAT_MISC'         , True)
        sp.STAT_REFLECTIONS   = kwargs.get('STAT_REFLECTIONS'  , True)
        sp.STAT_PHYSICS       = kwargs.get('STAT_PHYSICS'      , True)
        sp.STAT_XY_RESOLUTION = kwargs.get('STAT_XY_RESOLUTION', False) # not available for QUAD

        if sp.STAT_TIME_CH :
           sp.lst_u1 = []
           sp.lst_u2 = []
           sp.lst_v1 = []
           sp.lst_v2 = []
           sp.lst_w1 = []
           sp.lst_w2 = []
           sp.lst_mcp= []
            
        if sp.STAT_NHITS :
           sp.lst_nhits_u1 = []
           sp.lst_nhits_u2 = []
           sp.lst_nhits_v1 = []
           sp.lst_nhits_v2 = []
           sp.lst_nhits_w1 = []
           sp.lst_nhits_w2 = []
           sp.lst_nhits_mcp= []
           sp.lst_nparts   = []

        if sp.STAT_UVW or sp.STAT_CORRELATIONS :
           sp.lst_u_ns = []
           sp.lst_v_ns = []
           sp.lst_w_ns = []
           sp.lst_u = []
           sp.lst_v = []
           sp.lst_w = []
        
        if sp.STAT_TIME_SUMS or sp.STAT_CORRELATIONS :
           sp.lst_time_sum_u = []
           sp.lst_time_sum_v = []
           sp.lst_time_sum_w = []
           
           sp.lst_time_sum_u_corr = []
           sp.lst_time_sum_v_corr = []
           sp.lst_time_sum_w_corr = []

        if sp.STAT_XY_COMPONENTS :
           sp.lst_Xuv = []
           sp.lst_Xuw = []
           sp.lst_Xvw = []
           sp.lst_Yuv = []
           sp.lst_Yuw = []
           sp.lst_Yvw = []

        if sp.STAT_MISC :
           sp.list_dr = []
           sp.lst_consist_indicator = []
           sp.lst_rec_method = []

        if sp.STAT_XY_RESOLUTION :
           sp.lst_binx = []
           sp.lst_biny = []
           sp.lst_resol_fwhm = []

        if sp.STAT_REFLECTIONS :
           sp.lst_refl_u1 = []
           sp.lst_refl_u2 = []
           sp.lst_refl_v1 = []
           sp.lst_refl_v2 = []
           sp.lst_refl_w1 = []
           sp.lst_refl_w2 = []

        if sp.STAT_XY_2D :
           # images 
           nbins = 360
           sp.img_x_bins = HBins((-45., 45.), nbins, vtype=np.float32)
           sp.img_y_bins = HBins((-45., 45.), nbins, vtype=np.float32)
           sp.img_xy_uv = np.zeros((nbins, nbins), dtype=np.float32)
           sp.img_xy_uw = np.zeros((nbins, nbins), dtype=np.float32)
           sp.img_xy_vw = np.zeros((nbins, nbins), dtype=np.float32)
           sp.img_xy_1  = np.zeros((nbins, nbins), dtype=np.float32)
           sp.img_xy_2  = np.zeros((nbins, nbins), dtype=np.float32)

        if sp.STAT_PHYSICS :
           t_ns_nbins = 300
           sp.t_ns_bins = HBins((1400., 2900.), t_ns_nbins, vtype=np.float32)
           #sp.t1_vs_t0 = np.zeros((t_ns_nbins, t_ns_nbins), dtype=np.float32)

           sp.ti_vs_tj = np.zeros((t_ns_nbins, t_ns_nbins), dtype=np.float32)
           sp.t_all = np.zeros((t_ns_nbins,), dtype=np.float32)
           sp.lst_t_all = []

           x_mm_nbins = 200
           y_mm_nbins = 200
           r_mm_nbins = 200
           sp.x_mm_bins = HBins((-50., 50.), x_mm_nbins, vtype=np.float32)
           sp.y_mm_bins = HBins((-50., 50.), y_mm_nbins, vtype=np.float32)
           sp.r_mm_bins = HBins((-50., 50.), r_mm_nbins, vtype=np.float32)
           #sp.x_vs_t0 = np.zeros((x_mm_nbins, t_ns_nbins), dtype=np.float32)
           #sp.y_vs_t0 = np.zeros((y_mm_nbins, t_ns_nbins), dtype=np.float32)
           sp.rsx_vs_t  = np.zeros((r_mm_nbins, t_ns_nbins), dtype=np.float32)
           sp.rsy_vs_t  = np.zeros((r_mm_nbins, t_ns_nbins), dtype=np.float32)

#----------

    def fill_raw_data(sp, number_of_hits, tdc_ns) :

        sorter = sp.sorter
        Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp = sorter.channel_indexes

        if sp.STAT_NHITS :
           sp.lst_nhits_mcp.append(number_of_hits[Cmcp])
           sp.lst_nhits_u1. append(number_of_hits[Cu1])
           sp.lst_nhits_u2 .append(number_of_hits[Cu2])
           sp.lst_nhits_v1 .append(number_of_hits[Cv1])
           sp.lst_nhits_v2 .append(number_of_hits[Cv2])
           if sorter.use_hex :        
               sp.lst_nhits_w1 .append(number_of_hits[Cw1])
               sp.lst_nhits_w2 .append(number_of_hits[Cw2])


        if sp.STAT_TIME_CH :
           sp.lst_mcp.append(tdc_ns[Cmcp,0])
           sp.lst_u1 .append(tdc_ns[Cu1,0])
           sp.lst_u2 .append(tdc_ns[Cu2,0])
           sp.lst_v1 .append(tdc_ns[Cv1,0])
           sp.lst_v2 .append(tdc_ns[Cv2,0])
           if sorter.use_hex :        
               sp.lst_w1 .append(tdc_ns[Cw1,0])
               sp.lst_w2 .append(tdc_ns[Cw2,0])

        if sp.STAT_REFLECTIONS :
           if number_of_hits[Cu2]>1 : sp.lst_refl_u1.append(tdc_ns[Cu2,1] - tdc_ns[Cu1,0])
           if number_of_hits[Cu1]>1 : sp.lst_refl_u2.append(tdc_ns[Cu1,1] - tdc_ns[Cu2,0])
           if number_of_hits[Cv2]>1 : sp.lst_refl_v1.append(tdc_ns[Cv2,1] - tdc_ns[Cv1,0])
           if number_of_hits[Cv1]>1 : sp.lst_refl_v2.append(tdc_ns[Cv1,1] - tdc_ns[Cv2,0])
           if sorter.use_hex :        
               if number_of_hits[Cw2]>1 : sp.lst_refl_w1.append(tdc_ns[Cw2,1] - tdc_ns[Cw1,0])
               if number_of_hits[Cw1]>1 : sp.lst_refl_w2.append(tdc_ns[Cw1,1] - tdc_ns[Cw2,0])

        time_sum_u = tdc_ns[Cu1,0] + tdc_ns[Cu2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_v = tdc_ns[Cv1,0] + tdc_ns[Cv2,0] - 2*tdc_ns[Cmcp,0]
        time_sum_w = tdc_ns[Cw1,0] + tdc_ns[Cw2,0] - 2*tdc_ns[Cmcp,0] if sorter.use_hex else 0

        logger.info("RAW time_sum_u: %.2f _v: %.2f _w: %.2f " % (time_sum_u, time_sum_v, time_sum_w))

        if sp.STAT_TIME_SUMS or sp.STAT_CORRELATIONS :
           sp.lst_time_sum_u.append(time_sum_u)
           sp.lst_time_sum_v.append(time_sum_v)
           sp.lst_time_sum_w.append(time_sum_w)


        #if sp.STAT_XY_RESOLUTION :
        # NOT VALID FOR QUAD
        #sfco = hexanode.py_scalefactors_calibration_class(sorter) # NOT FOR QUAD
        #    #logger.info("    binx: %d  biny: %d  resolution(FWHM): %.6f" % (sfco.binx, sfco.biny, sfco.detector_map_resol_FWHM_fill))
        #    if sfco.binx>=0 and sfco.biny>=0 :
        #        sp.lst_binx.append(sfco.binx)
        #        sp.lst_biny.append(sfco.biny)
        #        sp.lst_resol_fwhm.append(sfco.detector_map_resol_FWHM_fill)


#----------

    def fill_corrected_data(sp) :

        sorter = sp.sorter
        Cu1, Cu2, Cv1, Cv2, Cw1, Cw2, Cmcp = sorter.channel_indexes
        number_of_particles = sorter.output_number_of_hits

        if sp.STAT_NHITS :
           sp.lst_nparts.append(number_of_particles)

        # Discards most of events in command>1
        #=====================
        if number_of_particles<1 :
            logger.debug('no hits found in event ')
            return
        #=====================

        u_ns = tdc_ns[Cu1,0] - tdc_ns[Cu2,0]
        v_ns = tdc_ns[Cv1,0] - tdc_ns[Cv2,0]
        w_ns = 0 #tdc_ns[Cw1,0] - tdc_ns[Cw2,0]

        u = u_ns * sorter.fu
        v = v_ns * sorter.fv
        w = 0 #(w_ns + sp.w_offset) * sorter.fw

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
            w = (w_ns + sp.w_offset) * sorter.fw
            
            Xuw = u
            Xvw = v + w
            Yuv = (u - 2*v)*OSQRT3
            Yuw = (2*w - u)*OSQRT3
            Yvw = (w - v)*OSQRT3
            
            dX = Xuv - Xvw
            dY = Yuv - Yvw
            
            time_sum_w_corr = tdc_ns[Cw1,0] + tdc_ns[Cw2,0] - 2*tdc_ns[Cmcp,0]

        #---------

        if sp.STAT_UVW or sp.STAT_CORRELATIONS :
           sp.lst_u_ns.append(u_ns)
           sp.lst_v_ns.append(v_ns)
           sp.lst_w_ns.append(w_ns)
           sp.lst_u.append(u)
           sp.lst_v.append(v)
           sp.lst_w.append(w)

        if sp.STAT_TIME_SUMS or sp.STAT_CORRELATIONS :
           sp.lst_time_sum_u_corr.append(time_sum_u_corr)
           sp.lst_time_sum_v_corr.append(time_sum_v_corr)
           sp.lst_time_sum_w_corr.append(time_sum_w_corr)

        if sp.STAT_XY_COMPONENTS :
           sp.lst_Xuv.append(Xuv)
           sp.lst_Xuw.append(Xuw)
           sp.lst_Xvw.append(Xvw)
                            
           sp.lst_Yuv.append(Yuv)
           sp.lst_Yuw.append(Yuw)
           sp.lst_Yvw.append(Yvw)

        hco = py_hit_class(sorter, 0)

        if sp.STAT_MISC :
           dR = sqrt(dX*dX + dY*dY)
           sp.list_dr.append(dR)
           
           # fill Consistence Indicator
           consistenceIndicator = 0
           for (ind, incr) in sp.inds_incr :
             if number_of_hits[ind]>0 : consistenceIndicator += incr
           sp.lst_consist_indicator.append(consistenceIndicator)

           sp.lst_rec_method.append(hco.method)
           #logger.info('reconstruction method %d' % hco.method)


        if sp.STAT_XY_2D :

           # fill 2-d images
           x1, y1 = hco.x, hco.y

           x2, y2 = (-10,-10) 
           if number_of_particles > 1 :
               hco2 = py_hit_class(sorter, 1)
               x2, y2 = hco2.x, hco2.y

           ix1, ix2, ixuv, ixuw, ixvw = sp.img_x_bins.bin_indexes((x1, x2, Xuv, Xuw, Xvw))
           iy1, iy2, iyuv, iyuw, iyvw = sp.img_y_bins.bin_indexes((y1, y2, Yuv, Yuw, Yvw))

           sp.img_xy_1 [iy1,  ix1]  += 1
           sp.img_xy_2 [iy2,  ix2]  += 1
           sp.img_xy_uv[iyuv, ixuv] += 1
           sp.img_xy_uw[iyuw, ixuw] += 1 
           sp.img_xy_vw[iyvw, ixvw] += 1 

        if sp.STAT_PHYSICS :
          if number_of_hits[Cmcp]>1 :
            #t0, t1 = tdc_ns[Cmcp,:2]
            #it0, it1 = sp.t_ns_bins.bin_indexes((t0, t1))
            #sp.t1_vs_t0[it1, it0] += 1

            #ix, iy = sp.x_mm_bins.bin_indexes((Xuv,Yuv))
            #sp.x_vs_t0[ix, it0] += 1
            #sp.y_vs_t0[iy, it0] += 1

            #logger.info("  Event %5i  number_of_particles: %i" % (evnum, number_of_particles))
            #for i in range(number_of_particles) :
            #    hco = py_hit_class(sorter, i) 
            #    #logger.info("    p:%2i x:%7.3f y:%7.3f t:%7.3f met:%d" % (i, hco.x, hco.y, hco.time, hco.method))
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

#----------

if __name__ == "__main__" :

    print('%s\n%s'%(50*'_', USAGE))
    #o = DLDStatistics() 

    #=====================
    #sys.exit('TEST EXIT')
    #=====================

#----------

#----------
