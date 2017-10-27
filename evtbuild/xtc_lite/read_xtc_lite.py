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

mb_per_img = 16
batches = 16
arr_size = mb_per_img*500000
file_name = '/ztest/eliseo/data/xtc_lite/xtc_lite_%i.xtc' % rank

def read_client():

#    file_name = '../neh_dir/xtc_lite/xtc_lite.xtc'
 

    with open(file_name, 'rb') as f:
        f.seek(0)
        
        ct = 0
#        read_bytes = 0
        while True:
            file_info = os.stat(file_name)
            size_file_mb = os.stat(file_name).st_size/10**6

            # if ct + 1000 > size_file_mb:
            #     print('Near end of file. Waiting for more data')
            #     while True:
            #         time.sleep(0.1)
            #         size_file_mb = os.stat(file_name).st_size/10**6
                
            #         if ct + 2000 < size_file_mb:
            #             break

            img = f.read(mb_per_img*batches*10**6)
            ct+=1
            if ct%100 == 0:
                print('Read image %i' % ct)
            if img == '':
                break

comm.Barrier()
if rank == 0:
    size_file_mb = os.stat(file_name).st_size/10**6
    global_start = time.time()
    
read_client()

comm.Barrier()
if rank == 0:
    global_end = time.time()
    wrt_gb = size*size_file_mb/1000
    av_spd = wrt_gb/(global_end-global_start)
    print('Number of clients %i' % (size))
    print('File size %i' % wrt_gb) 
    print('Read %.2f GB at an average of %.2f GB/s' % (wrt_gb, av_spd))
