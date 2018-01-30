from mpi4py import MPI
import h5py, subprocess, glob, os
import numpy as np
import time

#logistical MPI setup
#assert size>1, 'At least 2 MPI ranks required'


import os


def read_client(comm):

    rank = comm.Get_rank()
    size = comm.Get_size()

    mb_per_img = 1
    batches = 16
    arr_size = mb_per_img*500000
    path = '/drpffb/eliseo/data/xtc_lite/'
    file_name = path + 'xtc_lite_%i.xtc' % rank

    # Check if the file exists
    while True:
        if os.path.exists(file_name):
            break
        else:
            time.sleep(0.05)
    

    with open(file_name, 'rb') as f:
        f.seek(0)
        
        ct = 0
        eof_pad = 2000

        while True:
            file_info = os.stat(file_name)
            size_file_mb = os.stat(file_name).st_size/10**6

            if ct + 1000 > size_file_mb and size_file_mb < 19000:
#                print('Near end of file. Waiting for more data')
                while True:
                    time.sleep(0.1)
                    size_file_mb = os.stat(file_name).st_size/10**6
                
                    if ct + eof_pad < size_file_mb:
                        break

            img = f.read(mb_per_img*batches*10**6)
            ct+=1
            if ct%100 == 0:
                pass
#                print('Read image %i' % ct)
            if img == '':
                break

def do_read(comm):
   # comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mb_per_img = 1
    batches = 16
    arr_size = mb_per_img*500000
    path = '/drpffb/eliseo/data/xtc_lite/'
    file_name = path + 'xtc_lite_%i.xtc' % rank


    comm.Barrier()
    if rank == 0:
       # 
        global_start = time.time()

    read_client(comm)

    comm.Barrier()
    if rank == 0:
        size_file_mb = os.stat(file_name).st_size/10**6
        global_end = time.time()
        wrt_gb = size*size_file_mb/1000
        av_spd = wrt_gb/(global_end-global_start)
        print('Finished at %s' % time.strftime("%H:%M:%S"))
        print('Number of clients %i' % (size))
        print('File size %i' % wrt_gb) 
        print('Copied %.2f GB at an average of %.2f GB/s' % (wrt_gb, av_spd))
        
