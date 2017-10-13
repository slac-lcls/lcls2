'''
New Filecreate file.
  timestamp length will now match big data length
big data array will be filled with integers instead of zeros
'''

# logistical support
#import h5py_cache
import h5py, random, sys, os,glob, subprocess, shutil
import numpy as np
from picklecreate import create_json, load_config

try:
        from tqdm import tqdm
        tqdm_exists = True
except ImportError:
        tqdm_exists = False

chunk_size = 1

#set number of stripes and the size
nstripes = 1
str_size = 1
start_index = 1

global nevents
nevents = int(sys.argv[1])
scr_dir = str(nevents/1000)+'k/'

config_dict = load_config()
n_hdf = config_dict['num_hdf']

scratch_path = config_dict['path']

path = scratch_path + '/'+scr_dir+'1Mblock_nstripes_'+str(nstripes)+'/'


#for i in range(n_hdf):
try:
        shutil.rmtree(path)
except OSError:
        pass

os.makedirs(path)



#Constant array. File 0 has all data, files >1 have even/odd events
def dist_tiles0(num_tiles, num_hdf,x):
	even_or_odd = x%2
      #  even_or_odd = 
	evt_dist = [1]+[(x+even_or_odd)%2 for x in range(num_hdf)]
      #  evt_dist =  [1]*(num_hdf+1)
	return evt_dist			


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

def dist_tiles(num_tiles, num_hdf,x):
#	evt_dist = np.random.choice(np.arange(num_hdf), size=8)
        evt_dist = np.random.multinomial(num_tiles, [1/float(num_hdf)]*num_hdf, size=1)
        return evt_dist[0]
#	return np.ndarray.tolist(evt_dist)

#create the small data in the first file
#@profile
def write_smalldata(nevents):
        str_comm = 'lfs setstripe %sfile0.h5 -c %i -S %iM -i %i' % (path,nstripes, str_size,start_index)
        print(str_comm)
        subprocess.call(str_comm, shell=True)

        with h5py.File('%sfile0.h5' % path, 'w', libver='latest') as f:
#         with h5py_cache.File('%sfile0.h5' % path, 'w',chunk_cache_mem_size=cache_size*1024**2) as f:
		small_data_group = f.create_group("small_data")
		diode_vals = small_data_group.create_dataset('diode_values', (nevents,), dtype='S5')
		all_timestamps = small_data_group.create_dataset('all_timestamps', (nevents,), dtype='i')
		 	
		all_timestamps[:] = range(nevents)
		
		diode_vals[:] = np.random.choice(smallDatColors,nevents)

#write out the data for the rest of the files
def write_file_output(file_num,exs,image_data):
	
	mode = 'w'
	if file_num ==0:
		mode = 'a'
	else:
                str_comm = 'lfs setstripe -c %i -S %iM -i %i %sfile%i.h5' % (nstripes, str_size,start_index+file_num,path,file_num)
                print(str_comm)
                subprocess.call(str_comm, shell=True)

	with h5py.File('%sfile%i.h5' % (path,file_num), mode, libver='latest') as f:
#	with h5py_cache.File('%sfile%i.h5' % (path,file_num), mode,chunk_cache_mem_size=cache_size*1024**2) as f:	
		print('Creating %sfile%i.h5' % (path,file_num))
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
		arraySizes = cspad_data_group.create_dataset('array_sizes', (num_evts,), dtype = 'uint16')
		arraySizes[:] = nonzero_evt_tiles
                rav_size = len(cspad_quad_rav)
                #rav_size = 143560
                ncopies = 2
                im_size = rav_size*ncopies
             #   image_data = cspad_data_group.create_dataset('image_data', shape = (num_evts,im_size), maxshape=(None,im_size),chunks = True, dtype = 'uint16')
		image_data = cspad_data_group.create_dataset('image_data', shape = (num_evts,im_size), maxshape=(None,im_size),chunks = (chunk_size,im_size), dtype = 'uint16')
                ev_ch_size = 100
                for ev_ch in range(0,num_evts,ev_ch_size):
                        image_data[ev_ch:ev_ch+ev_ch_size] = np.array([np.concatenate((image_dat_arr[0], np.tile(image_dat_arr[0],(ncopies-1))))]*ev_ch_size)

	#	image_data[:] = np.array([image_dat_arr[0]]*num_evts)

#small data will be an array of random colors. "red" will be deemed as an interesting event.
smallDatColors = ['red', 'blue', 'green']
#smallDatColors = ['red']
#distribute the tiles across the available hdf files
exs = np.asarray([dist_tiles0(8,n_hdf,x) for x in range(nevents)])

#make some mock cspad arrays and fill with random data
cspad_quad = np.random.randint(0,2**16,size=(250,250,4), dtype='uint16')
cspad_quad_rav = cspad_quad.ravel()
#create lookup table for different number of tiles
image_dat_arr = [np.r_[((cspad_quad_rav),)*nt] for nt in range(1,n_hdf+1)]


write_smalldata(nevents)
#
if tqdm_exists:
        for i in tqdm(range(n_hdf)):
                write_file_output(i,exs,image_dat_arr)
else:
        for i in range(n_hdf):
                write_file_output(i,exs,image_dat_arr)

print('Creating pickle')
create_json(path)
create_json(path, True)
print('Done')
