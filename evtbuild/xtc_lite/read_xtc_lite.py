from mpi4py import MPI
import h5py, subprocess, glob, os
import numpy as np
import time
import os, random

from load_config import load_config


cfg = load_config('sconfig')
hit_prob = float(cfg['hit_probability'])
write_limit = int(cfg['file_size'])
mb_per_img = int(cfg['image_size'])
batch_size = int(cfg['batch_size'])
bytes_per_batch = mb_per_img*batch_size*10**6


if int(cfg['parallel_disks']):
    disk_num = rank
else:
    disk_num = int(cfg['disk_num'])
    
path = cfg['path'] % disk_num


def read_client(comm,filt=0):

    rank = comm.Get_rank()
    size = comm.Get_size()

    # mb_per_img = 1
    # batches = 16
    arr_size = mb_per_img*500000
    # path = '/drpffb/eliseo/data/xtc_lite/'
   # file_name = path % rank + '/xtc_lite_%i.xtc' % rank

    file_name = path + '/xtc_lite_%i.xtc' % rank

    # Check if the file exists
    while True:
        if os.path.exists(file_name):
            break
        else:
            time.sleep(0.05)

    open_flags = (os.O_NONBLOCK | os.O_RDONLY)

    read_file = os.open(file_name, open_flags)
    os.lseek(read_file,0,0)

    ct = 0
    eof_pad = 20

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
        if not filt:
            img = os.read(read_file,bytes_per_batch)
        else:
            # Filter section
            # If RNG is below threshold, then read an image batch
            # else seek ahead.
            if(random.random()<hit_prob):
                img = os.read(read_file,bytes_per_batch)
            else:
                os.lseek(read_file,bytes_per_batch,os.SEEK_CUR)
                img='fff'
        ct+=1
        if ct%100 == 0:
            pass
        #                print('Read image %i' % ct)
        if img == '' and size_file_mb/1000 == write_limit:
            break
    os.close(read_file)

def do_read(comm, filt=0):
   # comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if filt:
        copy_word = "Filtered"
    else:
        copy_word = "Copied"
        hit_prob = 1
    # mb_per_img = 1
    # batch_size = 16
    arr_size = mb_per_img*500000
    # path = '/drpffb/eliseo/data/xtc_lite/'
    file_name = path  + '/xtc_lite_%i.xtc' % rank


    comm.Barrier()
    if rank == 0:
       #
        global_start = time.time()

    read_client(comm, filt)

    comm.Barrier()
    if rank == 0:
        size_file_mb = os.stat(file_name).st_size/10**6
        global_end = time.time()
        wrt_gb = hit_prob*size*size_file_mb/1000
        av_spd = wrt_gb/(global_end-global_start)
        print('\n'+'-'*40)
        print('Read completed at %s' % time.strftime("%H:%M:%S"))
        print('Number of clients %i' % (size))
        print('File size %i' % wrt_gb)
        print('%s %.2f GB at an average of %.2f GB/s' % (copy_word, wrt_gb, av_spd))
        print('-'*40+'\n')
