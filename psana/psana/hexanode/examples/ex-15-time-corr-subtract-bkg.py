#!/usr/bin/env python
##----------

import os
import sys
import numpy as np
from psana.pyalgos.generic.NDArrUtils import print_ndarr
from psana.pyalgos.generic.HBins import HBins
import psana.pyalgos.generic.Graphics as gr
import psana.pyalgos.generic.Utils as gu

#----------

def do_work() :

    prefix = './%s-figs-ti_vs_tj' % gu.str_tstamp(fmt='%Y-%m-%d', time_sec=None) # '%Y-%m-%dT%H:%M:%S%z'
    gu.create_directory(prefix, mode=0o775)

    path = os.path.abspath(os.path.dirname(__file__))
    print('path to npy flies dir:', path)

    ti_vs_tj = np.load('%s/ti_vs_tj.npy'%path)
    t_all    = np.load('%s/t_all.npy'%path)

    trange=(1400.,2900.)

    print_ndarr(ti_vs_tj, 'ti_vs_tj:\n')
    print_ndarr(t_all,    't_all:\n')

    sum_bkg = t_all.sum()
    sum_cor = ti_vs_tj.sum()

    print('sum_bkg:', sum_bkg)
    print('sum_cor:', sum_cor)

    imrange = trange + trange # (1400., 2900., 1400., 2900.)
    axim = gr.plotImageLarge(ti_vs_tj, img_range=imrange, amp_range=(0,500), figsize=(11,10),\
                             title='ti_vs_tj', origin='lower', window=(0.10, 0.08, 0.88, 0.88), cmap='inferno')
    gr.save('%s/fig-ti_vs_tj.png' % prefix)

    bkg = np.outer(t_all,t_all)/sum_bkg
    print_ndarr(bkg, 'bkg:\n')
    axim = gr.plotImageLarge(bkg, img_range=imrange, amp_range=(0,500), figsize=(11,10),\
                             title='bkg', origin='lower', window=(0.10, 0.08, 0.88, 0.88), cmap='inferno')
    gr.save('%s/fig-ti_vs_tj-bkg.png' % prefix)

    harr = t_all
    nbins = harr.size
    ht = HBins(trange, nbins, vtype=np.float32) # ht.binedges()
    fig, axhi, hi = gr.hist1d(ht.bincenters(), bins=nbins, amp_range=ht.limits(), weights=harr, color='b', show_stat=True,\
                              log=True, figsize=(7,6), axwin=(0.10, 0.10, 0.88, 0.85), title='1-d bkg',\
                              xlabel='time of all hits (ns)', ylabel='number of hits', titwin='1-d bkg')
    gr.save('%s/fig-time-hits.png' % prefix)
    gr.show()

#----------

if __name__ == "__main__" :

    print(50*'_')
    do_work()

#----------
