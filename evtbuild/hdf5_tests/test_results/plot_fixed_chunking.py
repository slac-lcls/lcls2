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
	
xtc_wr = [0.71, 0.71, 0.87, 0.79, 0.76, 0.74, 0.74, 0.8, 0.75, 0.75, 0.76, 0.75, 0.75, 0.75, 0.89, 0.74, 0.69, 0.66, 0.68, 0.75, 0.76, 0.75, 0.74, 0.74, 0.75, 0.74, 0.75, 0.77, 0.74, 0.74, 0.88, 0.91, 0.88, 0.77]
xtc_cp = [0.65, 0.65, 0.76, 0.66, 0.65, 0.65, 0.66, 0.65, 0.66, 0.66, 0.66, 0.65, 0.65, 0.66, 0.65, 0.65, 0.63, 0.62, 0.64, 0.65, 0.65, 0.65, 0.66, 0.66, 0.66, 0.66, 0.65, 0.66, 0.66, 0.66, 0.65, 0.75, 0.65, 0.64]

def plot_chunk_histo_write(num,fig,ax):
	filename = 'VL_1M_%i_chunking_write_3.txt' % num
	data_VL_wr = np.loadtxt(filename, skiprows=1)
	
	filename = 'HF_1M_%i_chunking_write_3.txt' % num
	data_HF_wr = np.loadtxt(filename, skiprows=1)
	
	filename = 'VL_1M_%i_chunking_read_3.txt' % num
	data_VL_re = np.loadtxt(filename, skiprows=1)
	
	filename = 'HF_1M_%i_chunking_read_3.txt' % num
	data_HF_re = np.loadtxt(filename, skiprows=1)
	
	bins = range(200,900,20)
	
	ax[0].set_title("Writing", fontsize=18)
	ax[0].hist(data_HF_wr[:,6],bins, label = "Fixed")
	ax[0].hist(data_VL_wr[:,6],bins, label = "Variable")
	ax[0].hist(1000*np.array(xtc_wr),bins, label = "Binary")

	#ax[0].axis([10,40,0,30])
	
	
	
	ax[1].set_title("Reading", fontsize=18)	

	ax[1].hist(data_HF_re[:,6],bins, label = "Fixed")
	ax[1].hist(data_VL_re[:,6], bins,label = "Variable")
	ax[1].hist(1000*np.array(xtc_cp),bins, label = "binary")

	#ax[1].axis([10,40,0,60])
#	ax[1].hist(data_VL[:,3]/1000)
#	ax[1].set_title("Variable length, %i element chunking" % num)
	ax[0].tick_params(labelsize=18)
	ax[1].tick_params(labelsize=18)
	
	ax[0].legend(loc="upper right", fontsize=20)
	ax[1].set_xlabel("Speed (MB/s)", fontsize=16)
	ax[1].set_ylabel("N", fontsize=16)

	ax[0].legend()
	fig.suptitle("10GB read/write for %i element chunking" % num, fontsize=20)
	fig.subplots_adjust(top=0.85)
	fig.savefig("%i_2xchunking_VL_HF_Bn_rw.png" % num)

def plot_read_comp(num,fig,ax):
	filename = 'VL_1M_%i_chunking_read2.txt' % num
	data_VL_re_2 = np.loadtxt(filename, skiprows=1)
	
	filename = 'HF_1M_%i_chunking_read2.txt' % num
	data_HF_re_2 = np.loadtxt(filename, skiprows=1)

	filename = 'VL_1M_%i_chunking_read_default.txt' % num
	data_VL_re_d = np.loadtxt(filename, skiprows=1)
	
	filename = 'HF_1M_%i_chunking_read_default.txt' % num
	data_HF_re_d = np.loadtxt(filename, skiprows=1)

	
	bins = range(200,900,20)
	
	ax[0].set_title("Default chunk cache")
	ax[0].hist(data_HF_re_d[:,6],bins, label = "Fixed")
	ax[0].hist(data_VL_re_d[:,6],bins, label = "Variable")
	
	ax[0].axvline(650, c='r', lw=2, ls='--')
	ax[1].axvline(650, c='r', lw=2, ls='--')	
	ax[0].annotate("fread()", xy=(630,30), color='red', rotation=90)
	#ax[0].axis([10,40,0,30])
	
	ax[1].set_title("2x chunk size cache")	

	ax[1].hist(data_HF_re_2[:,6],bins, label = "Fixed")
	ax[1].hist(data_VL_re_2[:,6], bins,label = "Variable")
	#ax[1].axis([10,40,0,60])
#	ax[1].hist(data_VL[:,3]/1000)
#	ax[1].set_title("Variable length, %i element chunking" % num)
	ax[0].legend(loc="upper right")
	ax[1].set_xlabel("Speed (MB/s)")
	ax[1].set_ylabel("N")
	
	ax[0].legend()
	fig.suptitle("10GB read for %i element chunking" % num, fontsize=20)
	fig.subplots_adjust(top=0.85)
	fig.savefig("%i_cache_VL_HF_read.png" % num)
	
#plot_read_comp(1024,fig,ax)
plot_chunk_histo_write(16,fig,ax)