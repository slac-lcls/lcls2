#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:51:03 2018

@author: tvk
"""

import matplotlib.pyplot as plt 
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#plt.style.use('ggplot')

if plt.get_fignums():
	for axe in ax:
		axe.clear()
else:	
	fig, ax = plt.subplots(3, sharex=True, figsize=(6*1.6,6))
		
		
def plot_HF_data(filename):
	fns = filename.split('_')
	
	data = np.loadtxt(filename, skiprows=1)
	

	ax[0].semilogx(data[:,0], data[:,6], 'C0',lw=3)
	ax[2].axhline(8, c='r',linestyle='--', lw=2)
	ax[1].semilogx(data[:,0], data[:,3]/1000, 'C0',lw=3)
	ax[2].semilogx(data[:,0], data[:,4], 'C0',lw=3)
	
	
	
	ax[2].set_xlabel('Chunk size (elements)', fontsize=14)
	ax[0].set_ylabel('Speed (MB/s)', fontsize=14)
	ax[1].set_ylabel('Duration (s)', fontsize=14)
	ax[2].set_ylabel('File size (MB)', fontsize=14)
	
	
	ax[0].tick_params(length=6, width=2, labelsize=14)
	ax[1].tick_params(length=6, width=2, labelsize=14)
	ax[2].tick_params(length=6, width=2, labelsize=14)
	

#	ax[0].axis([1,1.0*10**6,0,1.5])
#	ax[1].axis([1,1.0*10**6,0,60])
#	ax[2].axis([1,1.0*10**6,0,60])
		   

	plt.tight_layout()
	fig.suptitle('HDF test results for %s %s' % (fns[0], fns[-1][:-4]))
	fig.subplots_adjust(top=0.93)


	fig.savefig('HDF_%s_%s_test.pdf' % (fns[0], fns[-1][:-4]))
	
#plot_HF_data("HF_2bytes.txt")
plot_HF_data("VL_1M_write.txt")