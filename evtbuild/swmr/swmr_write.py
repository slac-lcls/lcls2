from mpi4py import MPI
import h5py, subprocess, glob, os
import numpy as np
import time
from load_config import load_config

#logistical MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#assert size>1, 'At least 2 MPI ranks required'

cfg = load_config('sconfig')
file_size = int(cfg['file_size'])
mb_per_img = cfg['image_size']
batch_size = int(cfg['batch_size'])
nstripes = int(cfg['nstripes'])

def create_image():
    image = np.random.randint(0,2**16,size=(int(250*250*8*mb_per_img)), dtype='uint16')
   # arr = np.tile(image,2*mb_per_img)
    arr = image.ravel()
    return arr

out_img = np.array([create_image() for x in range(batch_size)])

# Prepare all the files for writing and reading

file_name = cfg['path']+'/swmr_file%i.h5'

#create the files striped as above and prepared for swmr
def master():

    files = glob.glob(cfg['path']+'/swmr_file*')
    for fil in files:
        os.remove(fil)
 
    av_files=[]

    # Create the small data file
    small_file_name = cfg['path']+'/swmr_small_data.h5'
    small_file = h5py.File(small_file_name, 'w', libver='latest')
   
    small_data_group = small_file.create_group('small_data')
    diode_vals = small_data_group.create_dataset('diode_values', shape= (1,), chunks=(1,), maxshape=(None,),dtype='f')
    timestamps = small_data_group.create_dataset('timestamps', shape = (1,), chunks=(1,), maxshape=(None,), dtype='i')
    small_file.swmr_mode = True
    small_file.close()

    # Create the data files

    arr_size = mb_per_img*500000
    for fc in range(size):

        loop_fn = file_name % fc
        print('Creating %s' % loop_fn)
        str_comm = 'lfs setstripe -c %i -S 1M %s' % (nstripes, loop_fn)
       # print(str_comm)
        subprocess.call(str_comm, shell=True)

        f = h5py.File(loop_fn, 'w', libver='latest')
        dset = f.create_dataset("data", shape = (0,arr_size), chunks = (batch_size, arr_size), maxshape=(None,arr_size), dtype = 'uint16')

        small_data_group = f.create_group("small_data")        
        timestamps = small_data_group.create_dataset('timestamps', shape = (1,), chunks=(1,), maxshape=(None,), dtype='i')
        
        f.swmr_mode = True
        f.close()
    
def client():
  
    #Flag for determining if we've hit the file capacity limit
    eof = False
    written_mb = 0
    
    loop_fn = file_name % rank
   # print('Client %i, %s' %(rank, loop_fn))
    loop_file = h5py.File(loop_fn, 'a', libver='latest')
    loop_file.swmr_mode = True

    data_dset = loop_file['data']
    ts_dset = loop_file['small_data']['timestamps']

    if rank == 0:
        small_data_file = h5py.File(cfg['path']+'/swmr_small_data.h5', 'a', libver='latest')
        diode_dset = small_data_file['small_data']['diode_values']
        small_ts_dset = small_data_file['small_data']['timestamps']
        small_data_file.swmr_mode = True
        
    try:
        while True:
            start = time.time()
            shape = data_dset.shape
            batch_num = (shape[0]/batch_size) +1

            ts_dset.resize((shape[0]+batch_size,))


            data_dset.resize((shape[0]+batch_size, shape[1]))
       
            written_mb += out_img.nbytes/10**6

            # If we are rank 0, write out the diode values
            if rank == 0:
                diode_dset.resize((shape[0]+batch_size,)) 
                diode_dset[-batch_size:] = np.random.rand(batch_size)
                if written_mb > file_size*1000:
                    diode_dset[-1] = -1
                
                small_ts_dset.resize((shape[0]+batch_size,)) 
                small_ts_dset[-batch_size:] = np.arange(batch_size) + batch_size*(batch_num-1)
                small_data_file.flush()

                
            # Write the last image as all zeros as a flag to 
            # the readers that they've reached the end of the file
            if written_mb > file_size*1000:
                out_img[:] = 0
                print('Client %i has completed writing' % rank)
                eof = True

                
            # Else, write out a batch of random data and the timestamps
            loop_file['data'][-batch_size:,:] = out_img
            ts_dset[-batch_size:] = np.arange(batch_size) + batch_size*(batch_num-1)
   
            loop_file.flush()
       
            end = time.time()
            cr_speed = out_img.nbytes/(10**6*(end-start))
            
            if eof:
               # print('Client %i stopped writing' % rank)
                break
            image_num = batch_num*batch_size
        #    if image_num % 500 ==0:
                #print('Wrote to image %i at %i MB/s' % (batch_num*batch_size, cr_speed))
            
        

    finally:
        final_size = data_dset.shape
        loop_file.close()

    return final_size[0]

if rank ==0:
    master()

comm.Barrier()
if rank == 0:
    global_start = time.time()

rm = client()

comm.Barrier()
if rank == 0:
    global_end = time.time()
    elapsed_time = global_end - global_start
    wrt_gb = size*file_size
    av_spd = wrt_gb/elapsed_time

    print('Elapsed time %f s' % elapsed_time)
    print('Number of clients %i' % (size))
    print('Number of events %i' % rm)
    print('Number of stripes %i' % nstripes)
    print('Wrote %.2f GB at an average of %.2f GB/s' % (wrt_gb, av_spd))
