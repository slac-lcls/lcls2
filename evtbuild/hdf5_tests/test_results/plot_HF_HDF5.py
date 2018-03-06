#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:51:03 2018

@author: tvk
"""

import matplotlib.pyplot as plt 
import numpy as np
#plt.style.use('ggplot')



data = np.loadtxt("1M_evts_chunk_varied_2_ints.txt")

if plt.get_fignums():
	for axe in ax:
		axe.clear()
else:	
	fig, ax = plt.subplots(3, sharex=True)

#
#hdf_drplu_1stripe = proc_data('drp_two_nodes_1stripe.txt')
#xtc_drplu_1stripe = proc_data('xtc_two_nodes_1_stripe.txt')
#
#
#
ax[0].semilogx(10**6/data[:,0], data[:,6], 'C0',lw=3)
ax[2].axhline(8, c='r',linestyle='--', lw=2)
ax[1].semilogx(10**6/data[:,0], data[:,3]/1000, 'C0',lw=3)
ax[2].semilogx(10**6/data[:,0], data[:,5]*8, 'C0',lw=3)



ax[2].set_xlabel('Number of chunks', fontsize=20)
ax[0].set_ylabel('Speed (MB/s)', fontsize=20)
ax[1].set_ylabel('Duration (s)', fontsize=20)
ax[2].set_ylabel('File size (MB)', fontsize=20)


ax[0].tick_params(length=6, width=2, labelsize=16)
ax[1].tick_params(length=6, width=2, labelsize=16)
ax[2].tick_params(length=6, width=2, labelsize=16)


ax[0].axis([1,1.2*10**6,0,1.2])
ax[1].axis([1,1.2*10**6,0,60])
ax[2].axis([1,1.2*10**6,0,60])


plt.tight_layout()
#
#plt.legend()
fig.savefig('8MB_HDF_write_test.pdf')