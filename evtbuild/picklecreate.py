'''
This script is the second of the primitive offline event builder trilogy.
This is a single core script that matches the indices of all identical 
timestamps and stores them into an array which is dumped into a pickle
file for future analysis.
'''

import pickle, h5py, glob, sys
import numpy as np
from numba import jit, autojit
#import line_profiler

#this is how many h5 files the script has to run through
file_folder = str(sys.argv[1])

#runs = np.arange(8)

path = '/reg/d/psdm/cxi/cxitut13/scratch/eliseo/' +file_folder + '/'

#opens all the h5 files.


#@profile
def load_files():
    file_list = glob.glob(path+'*.h5')
    file_list = np.sort(file_list)
    files=[]
    for  filename in file_list:
        f = h5py.File(filename)
        files.append(f)
    return files



#@profile
def prepare_timestamps(files):
		
	all_ts_list = []
	#look for interesting events => truncated small data

	#truncSD = [i for i, x in enumerate(files[0]['smalldata']) if 'red' in x]
	#np.where is faster than list comprehension
	truncSD = np.where(np.array(files[0]['small_data/diode_values']) == 'red')
	truncSD = np.ndarray.tolist(truncSD[0])
	
	#pull timestamps of interesting events => truncated timestamps
#	truncTS = [files[0]['timestamp1'][i] for i in truncSD]
	#np.take is faster than list comprehension
	truncTS = np.take(np.array(files[0]['small_data/all_timestamps']), truncSD)	
	#Create a list of all the timestamps

	alt=[]
        file_lens = [len(truncTS)]

        file_lens =[]
        for file in files:
            line = file['small_data/time_stamp'][:]
            alt = np.concatenate((alt,line))
            file_lens.append(len(line))
        bins = np.cumsum(file_lens)

        return truncTS, bins, alt


# return a dictionary containing the file numbers and indices for a given
# event of interest

def find_timestamps3(alt, bins, truncTS):
    left_bins = np.concatenate(([0],bins))[:-1]
    
    def find_pos(x,y):
        ret=x[0] - np.take(left_bins, y)
        return ret

    flat_indexes = map(lambda x: np.where(alt == x), truncTS)
    array_inds = map(lambda x: np.digitize(x,bins)[0], flat_indexes)
    array_pos = map(find_pos, flat_indexes,array_inds)
  
    chunk_loc = dict(zip(truncTS, np.c_[array_inds,array_pos]))
    return chunk_loc


		
files = 	load_files()
truncTS, bins, alt = prepare_timestamps(files)
chunk_loc = find_timestamps3(alt, bins, truncTS)


file_Name = path+"eventpickle"
with open(file_Name, 'wb') as f:
	pickle.dump(chunk_loc, f)

[x.close for x in files]
