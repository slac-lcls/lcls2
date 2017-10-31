from mpi4py import MPI
import h5py, subprocess, glob, os
import numpy as np
import time

#logistical MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#assert size>1, 'At least 2 MPI ranks required'

# file size limit (GB)

file_size = 100


#1 MB images
mb_per_img = 16
batch_size = 16

arr_size = mb_per_img*500000
def create_image():
    image = np.random.randint(0,2**16,size=(250,250,8*mb_per_img), dtype='uint16')
   # arr = np.tile(image,2*mb_per_img)
    arr = image.ravel()
    return arr

out_img = np.array([create_image() for x in range(batch_size)])

#print(arr.shape[0])


def load_config():
    config_dict = {}
    with open("sconfig", 'r') as f:
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

cfg = load_config()


# Prepare all the files for writing and reading

file_count = size

fc=0
file_name = cfg['path']+'/swmr_file%i.h5'
#print(file_name)
nstripes=3
stripe_size = 1


#create the files striped as above and prepared for swmr
def master():
    files = glob.glob(cfg['path']+'/swmr_file*')
    for fil in files:
        os.remove(fil)
 
   
    av_files=[]
    for fc in range(file_count):
        loop_fn = file_name % fc

        str_comm = 'lfs setstripe -c %i -S %iM %s' % (nstripes, stripe_size,loop_fn)
       # print(str_comm)
 #       subprocess.call(str_comm, shell=True)


        f = h5py.File(loop_fn, 'w', libver='latest')
        dset = f.create_dataset("data", shape = (0,arr_size), chunks = (batch_size, arr_size), maxshape=(None,arr_size), dtype = 'uint16')
        f.swmr_mode = True
        f.close()

def client():
    fc = 0 
    
    #Flag for determining if we've hit the file capacity limit
    eof = False

    file_num = rank
    loop_fn = file_name % file_num
   # print('Client %i, %s' %(rank, loop_fn))
    loop_file = h5py.File(loop_fn, 'a', libver='latest')
    loop_file.swmr_mode = True
    written_mb = 0

    try:
        while True:
            start = time.time()
            shape = loop_file["data"].shape
            batch_num = (shape[0]/batch_size) +1


            loop_file['data'].resize((shape[0]+batch_size, shape[1]))

            
        #    print(out_img[:100].size)
            written_mb += out_img.nbytes/10**6

            if written_mb > file_size*1000:
                out_img[:] = 0
                eof = True

            loop_file['data'][-batch_size:,:] = out_img
            loop_file.flush()
       
            end = time.time()
            cr_speed = out_img.nbytes/(10**6*(end-start))
            
            if eof:
                break
     #       print('Wrote data batch %i at %i MB/s' % (batch_num, cr_speed))
 #           print('Speed %i MB/s' % cr_speed)


    finally:
        loop_file.close()







if rank ==0:
    master()

comm.Barrier()
if rank == 0:
    global_start = time.time()

client()

comm.Barrier()
if rank == 0:
    global_end = time.time()
    wrt_gb = size*file_size
    av_spd = wrt_gb/(global_end-global_start)
    print('Number of clients %i' % (size))
    print('File size %i' % wrt_gb) 
    prxint('Number of stripes %i' % nstripes)
    print('Wrote %.2f GB at an average of %.2f GB/s' % (wrt_gb, av_spd))
