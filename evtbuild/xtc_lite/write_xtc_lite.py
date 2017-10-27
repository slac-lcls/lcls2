from mpi4py import MPI
import h5py, subprocess, glob, os
import numpy as np
import time

#logistical MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#assert size>1, 'At least 2 MPI ranks required'


import os



write_limit = 100 #GB
mb_per_img = 16
batch_size = 16

arr_size = mb_per_img*500000
def create_image():
    image = np.random.randint(0,2**16,size=(250,250,8*mb_per_img), dtype='uint16')
   # arr = np.tile(image,2*mb_per_img)
    arr = image.ravel()
    return arr

out_img = np.array([create_image() for x in range(batch_size)])

img_mb = out_img.nbytes/10**6


def write_client():

   # file_name = '/drp_dir/eliseo/data/xtc_lite/xtc_lite_%i.xtc' % rank

    file_name = '/ztest/eliseo/data/xtc_lite/xtc_lite_%i.xtc' % rank
    try:
        os.remove(file_name)
    except OSError:
        pass

    with open(file_name, 'wb') as f:
        ct = 0 
        written_mb = 0
        while True:
            f.write(out_img)
            ct+=1
            written_mb += img_mb
            if ct%10 == 0:
                print('Wrote image %i' % ct)
            if written_mb > write_limit*1000:
                break
    

comm.Barrier()
if rank == 0:
    global_start = time.time()

write_client()

comm.Barrier()
if rank == 0:
    global_end = time.time()
    wrt_gb = size*write_limit
    av_spd = wrt_gb/(global_end-global_start)
    print('Number of clients %i' % (size))
    print('File size %i' % wrt_gb) 
    print('Wrote %.2f GB at an average of %.2f GB/s' % (wrt_gb, av_spd))



