'''
This script is the second of the primitive offline event builder trilogy.
This is a single core script that matches the indices of all identical 
timestamps and stores them into an array which is dumped into a pickle
file for future analysis.
'''

import cPickle as pickle
#import pickle
import h5py, glob, sys,json
import numpy as np
#from numba import jit, autojit
#import line_profiler


#opens all the h5 files.


#@profile
def load_files(path):

    file_list = glob.glob(path+'/*.h5')
    file_list = np.sort(file_list)
    files=[]
    for  filename in file_list:
        f = h5py.File(filename)
        files.append(f)
    return files



#@profile
def prepare_timestamps(files, all_files = False):
		
	all_ts_list = []
	#look for interesting events => truncated small data

	#truncSD = [i for i, x in enumerate(files[0]['smalldata']) if 'red' in x]
	#np.where is faster than list comprehension
        if not all_files:
            truncSD = np.where(np.array(files[0]['small_data/diode_values']) == 'red')
        else:
            truncSD = np.where(np.array(files[0]['small_data/diode_values']) != '0')
	
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

def find_timestamps3(alt, bins, truncTS, json=False):
    left_bins = np.concatenate(([0],bins))[:-1]
    
    def find_pos(x,y):
        ret=x[0] - np.take(left_bins, y)
        return ret

    flat_indexes = map(lambda x: np.where(alt == x), truncTS)
    array_inds = map(lambda x: np.digitize(x,bins)[0], flat_indexes)
    array_pos = map(find_pos, flat_indexes,array_inds)
    if not json:
        chunk_loc = dict(zip(truncTS, np.c_[array_inds,array_pos]))
    else:
        trunc = [str(x) for x in truncTS]
        out = [[np.ndarray.tolist(x),np.ndarray.tolist(y)] for x,y in zip(array_inds,array_pos)]
       # print(len(out))
        chunk_loc = dict(zip(trunc, out))
    return chunk_loc


def create_pickle(path, all_files = False, subset=False):
    files = load_files(path)
    if subset:
        files = [files[x] for x in subset]

    truncTS, bins, alt = prepare_timestamps(files, all_files)
    chunk_loc = find_timestamps3(alt, bins, truncTS)
    
    file_Name = path+"eventpickle"
    if all_files:
        file_Name +='_all'
    with open(file_Name, 'wb') as f:
	pickle.dump(chunk_loc, f)
    [x.close for x in files]

   # return chunk_loc


def create_json(path, all_files = False, subset = False):
    files = load_files(path)
    
    if subset:
        files = [files[x] for x in subset]
        subset_suff = '_subset'
    else:
        subset_suff = ''

    truncTS, bins, alt = prepare_timestamps(files, all_files)
    chunk_loc = find_timestamps3(alt, bins, truncTS, True)
    
    file_Name = path+"/eventpickle"+subset_suff
    if all_files:
        file_Name +='_all'
   # print(file_Name)
    with open(file_Name, 'wb') as f:
	json.dump(chunk_loc, f)
    [x.close for x in files]

def load_config():
    config_dict = {}
    with open("config", 'r') as f:
         for line in f:
             if line[0] in ('#', '\n'):
                continue
             (key, val) = line.split()
             try:
                     val = eval(val)
             except SyntaxError:
                     pass

             config_dict[key] = val
    return config_dict


#create_json('/global/homes/e/eliseo/lcls2/evtbuild/data/100k/nstripes_1')
#create_json('/global/homes/e/eliseo/lcls2/evtbuild/data/100k/nstripes_1',True)




#dep = {int(k):v for k,v in json.load(open('/reg/d/ffb/xpp/test/offline_eb/data/100k/eventpickle_all', 'r')).iteritems()}

