from mpi4py import MPI
import h5py, glob,os
import numpy as np,time
import argparse

#logistical MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

batch_size = 16
mb_per_batch = 16

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-r','--random', action='store', help = "Randomize reading", dest="random", default = 0)
args = parser.parse_args()


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




# Each client reads a single file.
# It checks the size and, if it is greater than the number
# of batches already read, reads the next chunk. 
# Otherwise it waits for new data to be written  
def client(file_list):
    file_name = '%s/swmr_file%i.h5' % (cfg['path'],rank)
    file_num = rank/len(file_list)

    with h5py.File(file_name, 'r', swmr=True) as f:
        dset = f["data"]
       # dset.id.refresh()
        file_cap = dset.shape[0]
        written_batches = file_cap/batch_size 
        read_mb = 0

        print('Reading %s' % file_name)
        batch_num = 0
        eof_sw = False
        exit_ctr = 0
        while True:
            # while True:
            #     start = time.time()
            #     if batch_num == written_batches:
            #         dset.id.refresh()
            #         file_cap = dset.shape[0]
            #         written_batches = file_cap/batch_size 
            #     if batch_num == written_batches:
            #         time.sleep(0.05)
            #     elif batch_num < written_batches:
            #         break
            # while True:
            #     start = time.time()
            #     dset.id.refresh()
            #     file_cap = dset.shape[0]
            #     written_batches = file_cap/batch_size 
            #     if written_batches > batch_num:
            #         break

#            file_info = os.stat(file_name)
            size_file = os.stat(file_name).st_size/10**6
            prev_size = size_file
            
         #   print(eof_sw)
            if ((read_mb + 2000) > size_file and not eof_sw):
                print('Near end of file. Waiting for more data')
                while True:
                    time.sleep(.1)
                    size_file = os.stat(file_name).st_size/10**6
               #     print('Read %i/%i' % (read_mb,size_file))
                    if (read_mb +4000) < size_file:
                        print('refresh')
                        dset.id.refresh()
                        file_cap = dset.shape[0]
                        written_batches = file_cap/batch_size 
                        break
             #       print(size_file, prev_size)

                    if size_file > prev_size:
                        prev_size == size_file
                    if size_file == prev_size:
                        exit_ctr +=1


                    if exit_ctr > 5:
                        eof_sw = True
                        dset.id.refresh()
                        break
                    
            if batch_num == written_batches:
                dset.id.refresh()
                file_cap = dset.shape[0]
                written_batches = file_cap/batch_size 
              
                
            start =time.time()    
         #   print(read_mb, size_file,batch_num, written_batches)
            crcio = dset[batch_num*batch_size:(batch_num+1)*batch_size][:]

           # print(crcio.size, batch_num)
          #  print(len(crcio[0]))
           
            cr_mbytes = crcio.nbytes/10**6
            end = time.time()
            sleep_time = 0.0*(end-start)
            time.sleep(sleep_time)
            cr_speed = cr_mbytes/((end-start)+sleep_time)
            
#            print('Reading data batch %i/%i at %i MB/s' %(batch_num, written_batches, cr_speed))
#            print('Read %i MB/s' % cr_speed) 
            batch_num += 1
            read_mb += cr_mbytes 
            # try:
            #     crcio[0]
            # except IndexError:
            #     print('Client %i is finished. Read %i GB' % (rank, read_mb/1000))
            #     return read_mb


            if np.sum(crcio[0][-1000:]) == 0:
                print(read_mb)
                print('Client %i is finished. Read %i GB' % (rank, read_mb/1000))
                return read_mb


cfg = load_config()
file_list = glob.glob('%s/swmr_file*' % cfg['path'])
clients_per_file = size/len(file_list)



comm.Barrier()
if rank == 0:
    global_start = time.time()

rm = client(file_list)

comm.Barrier()
if rank == 0:
    global_end = time.time()
    read_gb = float(size*rm)/1000
    av_spd = read_gb/(global_end-global_start)

    print('File size %i' % read_gb) 
    print('Number of clients %i' % (size))
    print('Read %.2f GB at an average of %.2f GB/s' % (read_gb, av_spd))
