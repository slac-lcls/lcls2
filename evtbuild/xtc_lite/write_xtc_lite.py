from mpi4py import MPI
import h5py, subprocess, glob, os
import numpy as np
import time, os
from load_config import load_config



cfg = load_config('sconfig')
write_limit = int(cfg['file_size'])
mb_per_img = int(cfg['image_size'])
batch_size = int(cfg['batch_size'])

arr_size = mb_per_img*500000

disk_num = int(cfg['disk_num'])
path = cfg['path'] % disk_num

def create_image():
    image = np.random.randint(0,2**16,size=(250,250,8*mb_per_img), dtype='uint16')
   # arr = np.tile(image,2*mb_per_img)
    arr = image.ravel()
    return arr


def write_client(comm, ind):
    rank = comm.Get_rank()

    # file_name = '/drp_dir/eliseo/data/xtc_lite/xtc_lite_%i.xtc' % rank

    out_img = np.array([create_image() for x in range(batch_size)])

    img_mb = out_img.nbytes/10**6
    out_img = out_img.tobytes()
    # path = '/drpffb/eliseo/data/xtc_lite/'

    # uncomment for writing to n disks

    file_name = path + '/%sxtc_lite_%i.xtc' % (ind, rank)

    try:
        os.remove(file_name)
    except OSError:
        pass

    # with open(file_name, 'wb') as f:
    ct = 0
    open_flags = (os.O_CREAT | os.O_WRONLY)
    f = os.open(file_name, open_flags)
    written_mb = 0
    while True:
        os.write(f, out_img)
        ct += 1
        written_mb += img_mb
        if ct%10 == 0:
            pass
        #  print('Wrote image %i' % ct)
        if written_mb > write_limit*1000:
            break
    os.close(f)

def do_write(comm, ind =''):
#    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    comm.Barrier()
    if rank == 0:
        # files = glob.glob(path+'/*.xtc')
        # for f in files:
        #     os.remove(f)
        global_start = time.time()

    write_client(comm, ind)

    comm.Barrier()
    if rank == 0:
        global_end = time.time()
        wrt_gb = size*write_limit
        av_spd = wrt_gb/(global_end-global_start)
        print('\n'+'-'*40)
        print('Write completed  at %s' % time.strftime("%H:%M:%S"))
        print('Number of clients %i' % (size))
        print('File size %i GB' % wrt_gb)
        print('Wrote %.2f GB at an average of %.2f GB/s' % (wrt_gb, av_spd))

        print('-'*40+'\n')
