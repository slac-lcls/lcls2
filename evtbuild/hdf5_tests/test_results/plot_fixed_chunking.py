#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 16:19:07 2018

@author: tvk
"""
import pdb
import matplotlib.pyplot as plt 
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#plt.style.use('ggplot')


if plt.get_fignums():
	for axe in np.ndarray.flatten(ax):
		axe.clear()
else:	
	fig, ax = plt.subplots(2, sharex=True, figsize=(6*1.6,6))		
	

def plot_chunk_histo_write(num,fig,ax):
	filename = 'VL_1M_%i_chunking_write.txt' % num
	data_VL_wr = np.loadtxt(filename, skiprows=1)
	
	filename = 'HF_1M_%i_chunking_write.txt' % num
	data_HF_wr = np.loadtxt(filename, skiprows=1)
	
	filename = 'VL_1M_%i_chunking_read.txt' % num
	data_VL_re = np.loadtxt(filename, skiprows=1)
	
	filename = 'HF_1M_%i_chunking_read.txt' % num
	data_HF_re = np.loadtxt(filename, skiprows=1)
	
	
	ax[0].set_title("Write duration")
	ax[0].hist(data_HF_wr[:,3]/1000, range(40))
	ax[0].hist(data_VL_wr[:,3]/1000, range(40))
	ax[0].axis([10,40,0,30])
	
	ax[1].set_title("Read duration")	
	ax[1].hist(data_HF_re[:,3]/1000,range(40))
	ax[1].hist(data_VL_re[:,3]/1000,range(40))
	ax[1].axis([10,40,0,60])
#	ax[1].hist(data_VL[:,3]/1000)
#	ax[1].set_title("Variable length, %i element chunking" % num)
#	
	ax[1].set_xlabel("Time (s)")
	ax[1].set_ylabel("N")

	fig.suptitle("10GB read/write for %i element chunking" % num, fontsize=20)
	fig.subplots_adjust(top=0.85)
	fig.savefig("%i_chunking_VL_HF_rw.pdf" % num)


plot_chunk_histo_write(16,fig,ax)