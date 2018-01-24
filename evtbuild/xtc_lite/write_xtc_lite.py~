from mpi4py import MPI
import h5py, subprocess, glob, os
import numpy as np
import time

#logistical MPI setup
#assert size>1, 'At least 2 MPI ranks required'


import os

write_limit = 20 #GB
mb_per_img = 1
batch_size = 16

arr_size = mb_per_img*500000


def create_image():
    image = np.random.randint(0,2**16,size=(250,250,8*mb_per_img), dtype='uint16')
   # arr = np.tile(image,2*mb_per_img)
    arr = image.ravel()
    return arr


def write_client(comm):
    rank = comm.Get_rank()

   # file_name = '/drp_dir/eliseo/data/xtc_lite/xtc_lite_%i.xtc' % rank

    
    out_img = np.array([create_image() for x in range(batch_size)])

    img_mb = out_img.nbytes/10**6

    path = '/drpffb/eliseo/data/xtc_lite/'
    file_name = path + 'xtc_lite_%i.xtc' % rank

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
                pass
              #  print('Wrote image %i' % ct)
            if written_mb > write_limit*1000:
                break
    
def do_write(comm):
#    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    comm.Barrier()
    if rank == 0:
        global_start = time.time()

    write_client(comm)

    comm.Barrier()
    if rank == 0:
        global_end = time.time()
        wrt_gb = size*write_limit
        av_spd = wrt_gb/(global_end-global_start)
        print('Number of clients %i' % (size))
        print('File size %i GB' % wrt_gb) 
        print('Wrote %.2f GB at an average of %.2f GB/s' % (wrt_gb, av_spd))



