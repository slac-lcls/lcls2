'''
New Filecreate file.
timestamp length will now match big data length
big data array will be filled with integers instead of zeros
'''

# variable length data
# core that looks ahead, decides based on small data whether to accumulate
# and passes big data


#logistical support
import h5py_cache
import h5py, random, sys, os
import numpy as np
from tqdm import tqdm

n_hdf = 8
#nruns = 1000
global nevents
nevents = int(sys.argv[1])
scr_dir = str(nevents/1000)+'k/'
 
scratch_path = '/reg/d/psdm/cxi/cxitut13/scratch/eliseo/'
path = scratch_path + scr_dir
try:
        os.mkdir(path)
except OSError:
        print('Directory %s already exists' % path) 




#32 tile variable size image
def var_image(ntiles):
	img_out = []
	img_dims = []
	for tiles in range(ntiles):
		var_img_shape = np.random.randint(1,194),np.random.randint(1,185)
		rand_img = np.random.randint(0,255,size=var_img_shape,dtype='uint16')
		img_out = np.r_[(img_out,rand_img.ravel())]
		img_dims.append(var_img_shape)
	return img_out, np.array(img_dims).ravel()

def dist_tiles(num_tiles, num_hdf):
#	evt_dist = np.random.choice(np.arange(num_hdf), size=8)
        evt_dist = np.random.multinomial(num_tiles, [1/float(num_hdf)]*num_hdf, size=1)
        return evt_dist[0]
#	return np.ndarray.tolist(evt_dist)

#create the small data in the first file
#@profile
def write_smalldata(nevents):
	with h5py.File('%sfile0.h5' % path, 'w') as f: 
		small_data_group = f.create_group("small_data")
		diode_vals = small_data_group.create_dataset('diode_values', (nevents,), dtype='S5')
		all_timestamps = small_data_group.create_dataset('all_timestamps', (nevents,), dtype='i')
		 	
		all_timestamps[:] = range(nevents)
		
		diode_vals[:] = np.random.choice(smallDatColors,nevents)

#write out the data for the rest of the files
#@profile
def write_file_output(file_num,exs,image_data):
	
	mode = 'w'
	if file_num ==0:
		mode = 'a'
		
	#with h5py.File('file%i.h5' % file_num, mode, libver='latest') as f:
	with h5py_cache.File('%sfile%i.h5' % (path,file_num), mode,chunk_cache_mem_size=1024*1024**2) as f:	
	
		first_list = exs[:,file_num]
		nonzero_evt_timestamps = np.nonzero(first_list)[0]
		nonzero_evt_tiles = np.take(first_list, nonzero_evt_timestamps)
		num_evts = len(nonzero_evt_timestamps)
		
		#write out the timestamps of the events where data is stored in this hdf
		small_data_group = f.require_group("small_data")		
		time_stamps = small_data_group.create_dataset('time_stamp', (num_evts,), dtype='i')
		time_stamps[:] = np.ndarray.tolist(nonzero_evt_timestamps)
		
		#create a variable length datatype for the HDF file
		dt = h5py.special_dtype(vlen=np.dtype('uint16'))

		#create a group for the mock cspad data
		cspad_data_group = f.create_group("cspad_data")
		#write some metadata on the array sizes
		arraySizes = cspad_data_group.create_dataset('array_sizes', (num_evts,), dtype = dt)
		arraySizes[:] = np.c_[nonzero_evt_tiles]

		image_data = cspad_data_group.create_dataset('image_data', shape = (num_evts,), maxshape=(None,),chunks = (143560,), dtype = dt)
		image_data[:] = map(lambda x: image_dat_arr[x-1], nonzero_evt_tiles)
		
		#create a group for the mock reduced cspad data. These have random shapes
		cspad_red_data_group = f.create_group("cspad_reduced_data")
		#write some metadata on the array sizes
		arraySizes = cspad_red_data_group.create_dataset('array_sizes', (num_evts,), dtype = dt)
		image_data = cspad_red_data_group.create_dataset('image_data', shape = (num_evts,), maxshape=(None,),chunks = (143560,), dtype = dt)

		var_img_out = np.array(map(lambda x: var_image(x), nonzero_evt_tiles))
		arraySizes[:] = var_img_out[:,1]
		image_data[:] = var_img_out[:,0]
		

#small data will be an array of random colors. "red" will be deemed as an interesting event.
smallDatColors = ['red', 'blue', 'green']

#distribute the tiles across the available hdf files
exs = np.asarray([dist_tiles(8,8) for x in range(nevents)])

#make some mock cspad arrays and fill with random data
cspad_quad = np.random.randint(0,2**16,size=(194,185,4), dtype='uint16')
cspad_quad_rav = cspad_quad.ravel()
#create lookup table for different number of tiles
image_dat_arr = [np.r_[((cspad_quad_rav),)*nt] for nt in range(1,n_hdf+1)]


write_smalldata(nevents)
#
for i in tqdm(range(8)):
	write_file_output(i,exs,image_dat_arr)
