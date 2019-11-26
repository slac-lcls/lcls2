#!/usr/bin/env python
"""
Module :py:class:`DLDGraphics` for MCP DLD detectors for COLTRIMS experiments
===============================================================================

    from psana.hexanode.DLDGraphics import DLDGraphics

    kwargs = {'STAT_NHITS':True,...}
    p = DLDProcessor()
    s = DLDStatistics(p, **kwargs)

    # event loop with statustics accumulation

    draw_plots(s, **kwargs)

Created on 2019-11-20 by Mikhail Dubrovin
"""
#----------

USAGE = 'Run example: python .../psana/hexanode/examples/ex-....py'

#----------

import logging
logger = logging.getLogger(__name__)

import numpy as np
#import psana.pyalgos.generic.Utils as gu
from psana.pyalgos.generic.NDArrUtils import print_ndarr
from psana.pyalgos.generic.Graphics import hist1d, show, move_fig, save_fig, move, save, plotImageLarge, plotGraph

#----------

def plot_image(img,\
        figsize=(11,10),\
        axwin=(0.10, 0.08, 0.88, 0.88),\
        cmap='inferno',\
        title='x-y image',\
        xlabel='x',\
        ylabel='y',\
        titwin=None,\
        fnm='img.png',\
        amp_range=None,\
        img_range=None,\
        origin='upper',\
        hwin_x0y0=(10,10),\
        prefix='plot',\
        do_save=False) :
    """draws figure with image
    """
    s = img.shape
    _img_range = (0, s[1], s[0], 0) if img_range is None else img_range
    imgnb = img[1:-2,1:-2]
    _amp_range = (0, imgnb.mean() + 4*imgnb.std()) if amp_range is None else amp_range
    #_amp_range = (0, 0.2*img.max())
    axim = plotImageLarge(img, img_range=_img_range, amp_range=_amp_range, figsize=figsize,\
                          title=title, origin=origin, window=axwin, cmap=cmap)
    axim.set_xlabel(xlabel, fontsize=18)
    axim.set_ylabel(ylabel, fontsize=18)
    axim.set_title(title,   fontsize=12)
    move(hwin_x0y0[0], hwin_x0y0[1])
    save('%s-%s' % (prefix, fnm), do_save)
    #show()

#----------

def h1d(hlst,\
        bins=None,\
        amp_range=None,\
        weights=None,\
        color=None,\
        show_stat=True,\
        log=False,\
        figsize=(6,5),\
        axwin=(0.15, 0.12, 0.78, 0.80),\
        title='Title',\
        xlabel='x',\
        ylabel='y',\
        titwin=None,\
        fnm='hist.png',\
        hwin_x0y0=(10,10),\
        prefix='plot',\
        do_save=False) :
    """draws figure with 1d- histogram
    """
    fig, axhi, hi = hist1d(np.array(hlst), bins, amp_range, weights, color, show_stat,\
                           log, figsize, axwin, title, xlabel, ylabel, titwin)
    move(hwin_x0y0[0], hwin_x0y0[1])
    save('%s-%s' % (prefix, fnm), do_save)
    return fig, axhi, hi

#----------

def plot_graph(x, y,\
        figsize=(7,6),\
        pfmt='r-',\
        lw=2,\
        xlimits=None,\
        ylimits=None,\
        title='py vs. px',\
        xlabel='px',\
        ylabel='py',\
        fnm='graph.png',\
        hwin_x0y0=(10,10),\
        prefix='plot',\
        do_save=False) :
    """draws figure with graph
    """
    fig, ax = plotGraph(x, y, figsize=figsize, pfmt=pfmt, lw=lw)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title,   fontsize=12)
    move(hwin_x0y0[0], hwin_x0y0[1])
    save('%s-%s' % (prefix, fnm), do_save)

#----------
#class DLDGraphics :
#    """ holds, fills, and provide access to statistical arrays for MCP DLD data processing
#    """
#    def __init__(self, stats, **kwargs) :
#        self.stats = stats
#        logger.info('In DLDGraphics, **kwargs: %s' % str(kwargs))
#    def draw_histograms(self, prefix='plot', do_save=True, hwin_x0y0=(0,0)) :
#        plot_histograms(self.stats, prefix, do_save, hwin_x0y0)
#        show()
#----------

def draw_plots(sp, prefix='plot', do_save=True, hwin_x0y0=(0,400)) :
    """Plots/saves histograms
    """
    #---------
    if sp.STAT_NHITS :
    #---------
        nbins = 16
        limits = (-0.5,15.5)
        is_log = True

        h1d(np.array(sp.lst_nhits_u1), bins=nbins, amp_range=limits, log=is_log,\
            title ='Number of hits U1', xlabel='Number of hits U1', ylabel='Events',\
            fnm='nhits_u1.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_nhits_u2), bins=nbins, amp_range=limits, log=is_log,\
            title ='Number of hits U2', xlabel='Number of hits U2', ylabel='Events',\
            fnm='nhits_u2.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_nhits_v1), bins=nbins, amp_range=limits, log=is_log,\
            title ='Number of hits V1', xlabel='Number of hits V1', ylabel='Events',\
            fnm='nhits_v1.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_nhits_v2), bins=nbins, amp_range=limits, log=is_log,\
            title ='Number of hits V2', xlabel='Number of hits V2', ylabel='Events',\
            fnm='nhits_v2.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_nhits_w1), bins=nbins, amp_range=limits, log=is_log,\
        #    title ='Number of hits W1', xlabel='Number of hits W1', ylabel='Events',\
        #    fnm='nhits_w1.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_nhits_w2), bins=nbins, amp_range=limits, log=is_log,\
        #    title ='Number of hits W2', xlabel='Number of hits W2', ylabel='Events',\
        #    fnm='nhits_w2.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_nhits_mcp), bins=nbins, amp_range=limits, log=is_log,\
            title ='Number of hits MCP', xlabel='Number of hits MCP', ylabel='Events',\
            fnm='nhits_mcp.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_nparts), bins=nbins, amp_range=limits, log=is_log,\
            title ='Number of particles', xlabel='Number of particles', ylabel='Events',\
            fnm='nparticles.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

    #---------
    if sp.STAT_TIME_CH :
    #---------
        nbins = 300
        limits = (1000,4000)
        #limits = (0,10000)

        #print_ndarr(sp.lst_u1, 'U1')
        h1d(np.array(sp.lst_u1), bins=nbins, amp_range=limits, log=True,\
            title ='Time U1', xlabel='U1 (ns)', ylabel='Events',\
            fnm='time_u1_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #print_ndarr(sp.lst_u2, 'U2')
        h1d(np.array(sp.lst_u2), bins=nbins, amp_range=limits, log=True,\
            title ='Time U2', xlabel='U2 (ns)', ylabel='Events',\
            fnm='time_u2_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_v1), bins=nbins, amp_range=limits, log=True,\
            title ='Time V1', xlabel='V1 (ns)', ylabel='Events',\
            fnm='time_v1_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_v2), bins=nbins, amp_range=limits, log=True,\
            title ='Time V2', xlabel='V2 (ns)', ylabel='Events',\
            fnm='time_v2_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_w1), bins=nbins, amp_range=limits, log=True,\
        #    title ='Time W1', xlabel='W1 (ns)', ylabel='Events',\
        #    fnm='time_w1_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_w2), bins=nbins, amp_range=limits, log=True,\
        #    title ='Time W2', xlabel='W2 (ns)', ylabel='Events',\
        #    fnm='time_w2_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #print_ndarr(sp.lst_mcp, 'MCP')
        h1d(np.array(sp.lst_mcp), bins=nbins, amp_range=limits, log=True,\
            title ='Time MCP', xlabel='MCP (ns)', ylabel='Events',\
            fnm='time_mcp_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

    #---------
    if sp.STAT_TIME_SUMS :
    #---------
        nbins = 200
        limits = (0,200) # (50,180)
        #nbins = 250
        #limits = (0,5000)

        #print_ndarr(sp.lst_time_sum_u, 'U')
        h1d(np.array(sp.lst_time_sum_u), bins=nbins, amp_range=limits, log=True,\
            title ='Time sum U', xlabel='Time sum U (ns)', ylabel='Events',\
            fnm='time_sum_u_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #print_ndarr(sp.lst_time_sum_v, 'V')
        h1d(np.array(sp.lst_time_sum_v), bins=nbins, amp_range=limits, log=True,\
            title ='Time sum V', xlabel='Time sum V (ns)', ylabel='Events',\
            fnm='time_sum_v_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #print_ndarr(sp.lst_time_sum_w, 'W')
        #h1d(np.array(sp.lst_time_sum_w), bins=nbins, amp_range=limits, log=True,\
        #    title ='Time sum W', xlabel='Time sum W (ns)', ylabel='Events',\
        #    fnm='time_sum_w_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

    #---------
    if sp.STAT_TIME_SUMS :
    #---------
        nbins = 160
        limits = (-80,80)
        h1d(np.array(sp.lst_time_sum_u_corr), bins=nbins, amp_range=limits, log=True,\
            title ='Time sum U corrected', xlabel='Time sum U (ns) corrected', ylabel='Events',\
            fnm='time_sum_u_ns_corr.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_time_sum_v_corr), bins=nbins, amp_range=limits, log=True,\
            title ='Time sum V corrected', xlabel='Time sum V (ns) corrected', ylabel='Events',\
            fnm='time_sum_v_ns_corr.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_time_sum_w_corr), bins=nbins, amp_range=limits, log=True,\
        #    title ='Time sum W corrected', xlabel='Time sum W (ns) corrected', ylabel='Events',\
        #    fnm='time_sum_w_ns_corr.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

    #---------
    if sp.STAT_UVW :
    #---------
        nbins = 200
        limits = (-100,100)

        h1d(np.array(sp.lst_u), bins=nbins, amp_range=limits, log=True,\
            title ='U (mm)', xlabel='U (mm)', ylabel='Events',\
            fnm='u_mm.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_v), bins=nbins, amp_range=limits, log=True,\
            title ='V (mm)', xlabel='V (mm)', ylabel='Events',\
            fnm='v_mm.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_w), bins=nbins, amp_range=limits, log=True,\
        #    title ='W (mm)', xlabel='W (mm)', ylabel='Events',\
        #    fnm='w_mm.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

    #---------
    if sp.STAT_UVW :
    #---------
        nbins = 300
        limits = (-150,150)

        h1d(np.array(sp.lst_u_ns), bins=nbins, amp_range=limits, log=True,\
            title ='U (ns)', xlabel='U (ns)', ylabel='Events',\
            fnm='u_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_v_ns), bins=nbins, amp_range=limits, log=True,\
            title ='V (ns)', xlabel='V (ns)', ylabel='Events',\
            fnm='v_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_w_ns), bins=nbins, amp_range=limits, log=True,\
        #    title ='W (ns)', xlabel='W (ns)', ylabel='Events',\
        #    fnm='w_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

    #---------
    if sp.STAT_CORRELATIONS :
    #---------
         #print_ndarr(sp.lst_time_sum_u, 'time_sum_u')
         #print_ndarr(sp.lst_u_ns,      'lst_u_ns ')
         xlimits=(-100,100)
         #ylimits=(20,120)
         ylimits=(50,180)

         plot_graph(sp.lst_u_ns, sp.lst_time_sum_u, figsize=(8,7), pfmt='b, ', lw=1, xlimits=xlimits, ylimits=ylimits,\
            title='t sum vs. U', xlabel='U (ns)', ylabel='t sum U (ns)',\
            fnm='t_sum_vs_u_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

         plot_graph(sp.lst_v_ns, sp.lst_time_sum_v, figsize=(8,7), pfmt='b, ', lw=1, xlimits=xlimits, ylimits=ylimits,\
            title='t sum vs. V', xlabel='V (ns)', ylabel='t sum V (ns)',\
            fnm='t_sum_vs_v_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

         #plot_graph(sp.lst_w_ns, sp.lst_time_sum_w, figsize=(8,7), pfmt='b, ', lw=1, xlimits=xlimits, ylimits=ylimits,\
         #   title='t sum vs. W', xlabel='W (ns)', ylabel='t sum W (ns)',\
         #   fnm='t_sum_vs_w_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

         #---------
         xlimits=(-100,100)
         ylimits=(-80,20)
         #---------
         plot_graph(sp.lst_u_ns, sp.lst_time_sum_u_corr, figsize=(8,7), pfmt='b, ', lw=1, xlimits=xlimits, ylimits=ylimits,\
            title='t sum corrected vs. U', xlabel='U (ns)', ylabel='t sum corrected U (ns)',\
            fnm='t_sum_corr_vs_u_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

         plot_graph(sp.lst_v_ns, sp.lst_time_sum_v_corr, figsize=(8,7), pfmt='b, ', lw=1, xlimits=xlimits, ylimits=ylimits,\
            title='t sum_corrected vs. V', xlabel='V (ns)', ylabel='t sum corrected V (ns)',\
            fnm='t_sum_corr_vs_v_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

         #plot_graph(sp.lst_w_ns, sp.lst_time_sum_w_corr, figsize=(8,7), pfmt='b, ', lw=1, xlimits=xlimits, ylimits=ylimits,\
         #   title='t sum_corrected vs. W', xlabel='W (ns)', ylabel='t sum corrected W (ns)',\
         #   fnm='t_sum_corr_vs_w_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

    #---------
    if sp.STAT_XY_COMPONENTS :
    #---------
        nbins = 200
        limits = (-50,50)

        h1d(np.array(sp.lst_Xuv), bins=nbins, amp_range=limits, log=True,\
            title ='Xuv', xlabel='Xuv (mm)', ylabel='Events',\
            fnm='Xuv_mm.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_Xuw), bins=nbins, amp_range=limits, log=True,\
        #    title ='Xuw', xlabel='Xuw (mm)', ylabel='Events',\
        #    fnm='Xuw_mm.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_Xvw), bins=nbins, amp_range=limits, log=True,\
        #    title ='Xvw', xlabel='Xvw (mm)', ylabel='Events',\
        #    fnm='Xvw_mm.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_Yuv), bins=nbins, amp_range=limits, log=True,\
            title ='Yuv', xlabel='Yuv (mm)', ylabel='Events',\
            fnm='Yuv_mm.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_Yuw), bins=nbins, amp_range=limits, log=True,\
        #    title ='Yuw', xlabel='Yuw (mm)', ylabel='Events',\
        #    fnm='Yuw_mm.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_Yvw), bins=nbins, amp_range=limits, log=True,\
        #    title ='Yvw', xlabel='Yvw (mm)', ylabel='Events',\
        #    fnm='Yvw_mm.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

    #---------
    if sp.STAT_REFLECTIONS :
    #---------
        #nbins = 150
        #limits = (-100, 5900)
        nbins = 300
        limits = (-500, 2500)

        h1d(np.array(sp.lst_refl_u1), bins=nbins, amp_range=limits, log=True,\
            title ='Reflection U1', xlabel='Reflection U1 (ns)', ylabel='Events',\
            fnm='refl_u1_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_refl_u2), bins=nbins, amp_range=limits, log=True,\
            title ='Reflection U2', xlabel='Reflection U2 (ns)', ylabel='Events',\
            fnm='refl_u2_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_refl_v1), bins=nbins, amp_range=limits, log=True,\
            title ='Reflection V1', xlabel='Reflection V1 (ns)', ylabel='Events',\
            fnm='refl_v1_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_refl_v2), bins=nbins, amp_range=limits, log=True,\
            title ='Reflection V2', xlabel='Reflection V2 (ns)', ylabel='Events',\
            fnm='refl_v2_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_refl_w1), bins=nbins, amp_range=limits, log=True,\
        #    title ='Reflection W1', xlabel='Reflection W1 (ns)', ylabel='Events',\
        #    fnm='refl_w1_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        #h1d(np.array(sp.lst_refl_w2), bins=nbins, amp_range=limits, log=True,\
        #    title ='Reflection W2', xlabel='Reflection W2 (ns)', ylabel='Events',\
        #    fnm='refl_w2_ns.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

    #---------
    if sp.STAT_MISC :
    #---------
        h1d(np.array(sp.list_dr), bins=160, amp_range=(0,40), log=True,\
            title ='Deviation', xlabel='Deviation (mm)', ylabel='Events',\
            fnm='deviation_mm.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_consist_indicator), bins=64, amp_range=(0,64), log=True,\
            title ='Consistence indicator', xlabel='Consistence indicator (bit)', ylabel='Events',\
            fnm='consistence_indicator.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        h1d(np.array(sp.lst_rec_method), bins=64, amp_range=(0,32), log=True,\
            title ='Reconstruction method', xlabel='Method id (bit)', ylabel='Events',\
            fnm='reconstruction_method.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

    #---------
    if sp.STAT_XY_2D :
    #---------
        amp_limits = (0,5)
        imrange=(sp.img_x_bins.vmin(), sp.img_x_bins.vmax(), sp.img_y_bins.vmax(), sp.img_y_bins.vmin())
        plot_image(sp.img_xy_uv, amp_range=amp_limits, img_range=imrange, fnm='xy_uv.png',\
                   title='XY_uv image', xlabel='x', ylabel='y', titwin='XY_uv image',\
                   hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)
        #plot_image(sp.img_xy_uw, amp_range=amp_limits, img_range=imrange, fnm='xy_uw.png',\
        #            title='XY_uw image', xlabel='x', ylabel='y', titwin='XY_uw image',\
        #           hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)
        #plot_image(sp.img_xy_vw, amp_range=amp_limits, img_range=imrange, fnm='xy_vw.png',\
        #            title='XY_vw image', xlabel='x', ylabel='y', titwin='XY_vw image',\
        #           hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)
        plot_image(sp.img_xy_1,  amp_range=amp_limits, img_range=imrange, fnm='xy_1.png',\
                   title='XY image hit1', xlabel='x', ylabel='y', titwin='XY image hit1',\
                   hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)
        plot_image(sp.img_xy_2,  amp_range=amp_limits, img_range=imrange, fnm='xy_2.png',\
                   title='XY image hit2', xlabel='x', ylabel='y', titwin='XY image hit2',\
                   hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

    #---------
    if sp.STAT_PHYSICS :
    #---------
            #sp.t_ns_bins = HBins((1400., 2900.), t_ns_nbins, vtype=np.float32)
        ht = sp.t_ns_bins
        amp_limits = (0,5)
        imrange=(ht.vmin(), ht.vmax(), ht.vmin(), ht.vmax())

        plot_image(sp.ti_vs_tj, amp_range=amp_limits, img_range=imrange, fnm='ti_vs_tj.png',\
                   title='ti vs tj correlations', xlabel='tj (ns)', ylabel='ti (ns)', titwin='PIPICO', origin='lower',\
                   hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        limits = ht.vmin(), ht.vmax()
        t_arr = np.array(sp.lst_t_all)
        h1d(t_arr, bins=ht.nbins(), amp_range=limits, log=True,\
            title ='time of all hits', xlabel='t_all (ns)', ylabel='Events',\
            fnm='t_all.png', hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        t_all = ht.bin_count(t_arr)

        sum_bkg = t_all.sum()
        sum_cor = sp.ti_vs_tj.sum()

        print('number of entries for 1-d ti   (bkg):', sum_bkg)
        print('number of entries for ti vs tj (cor):', sum_cor)

        bkg = np.outer(t_all,t_all)/sum_bkg
        print_ndarr(bkg, 'bkg:\n')

        plot_image(bkg, amp_range=amp_limits, img_range=imrange, fnm='t_corr_bkg.png',\
                   title='ti vs tj background', xlabel='tj (ns)', ylabel='ti (ns)', titwin='PIPICO', origin='lower',\
                   hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

        imrange=(ht.vmin(), ht.vmax(), sp.r_mm_bins.vmin(), sp.r_mm_bins.vmax())
        plot_image(sp.rsy_vs_t,  amp_range=amp_limits, img_range=imrange, fnm='rsy_vs_t.png',\
                   title='r*sign(y) vs t (All hits)', xlabel='t (ns)', ylabel='r*sign(y) (mm)', titwin='r vs t (All hits)',\
                   origin='lower', figsize=(12,5), axwin=(0.08, 0.10, 0.95, 0.84),\
                   hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)
    #=========
    #=========
    show()
    #=========
    #=========

    #---------
    #if sp.STAT_XY_RESOLUTION :
    #---------
    #    npa_binx = np.array(sp.lst_binx)
    #    npa_biny = np.array(sp.lst_biny)
    #    max_binx = npa_binx.max()
    #    max_biny = npa_biny.max()
    #    print('binx.min/max: %d %d' % (npa_binx.min(), max_binx))
    #    print('biny.min/max: %d %d' % (npa_biny.min(), max_biny))
    #    max_bins = max(max_binx, max_biny) + 1        
    #    sp.img_xy_res = np.zeros((max_bins, max_bins), dtype=np.float64)
    #    sp.img_xy_sta = np.zeros((max_bins, max_bins), dtype=np.int32)
    #    sp.img_xy_res[npa_biny, npa_binx] += sp.lst_resol_fwhm # np.maximum(arr_max, nda)
    #    sp.img_xy_res[npa_biny, npa_binx] += 1
    #    sp.img_xy_res /= np.maximum(sp.img_xy_sta,1)
    #    plot_image(sp.img_xy_res, amp_range=None, img_range=(0,max_bins, 0,max_bins),\
    #               fnm='xy_res.png', title='Resolution FWHM (mm)', xlabel='x bins', ylabel='y bins', titwin='Resolution FWHM',\
    #               hwin_x0y0=hwin_x0y0, prefix=prefix, do_save=do_save)

#----------
