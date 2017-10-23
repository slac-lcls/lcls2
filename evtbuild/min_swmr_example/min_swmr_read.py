import h5py, glob
import numpy,time

# Each client reads a single file.
# It checks the size and, if it is greater than the number
# of batches already read, reads the next chunk. 
# Otherwise it waits for new data to be written  

# Number of events to read at once
batch_size = 10




file_name = 'swmr_file.py'


with h5py.File(file_name, 'r', swmr=True) as f:
    dset = f["data"]
    file_cap = dset.shape[0]
    written_batches = file_cap/batch_size 

    print('Reading %s' % file_name)
    batch_num = 0

    # Main loop over the events in dset
    while True:
        # Check the current batch_num against the size of the dset
        # if batch_num is equal to the current size of the dset
        # refresh it
        # Otherwise break

        while True:
            start = time.time()
            if batch_num == written_batches:
                dset.id.refresh()
                file_cap = dset.shape[0]
                written_batches = file_cap/batch_size 
            if batch_num == written_batches:
                time.sleep(0.05)
            elif batch_num < written_batches:
                break
        # This block will just refresh dset constantly until it finds new data
        # while True:
        #     start = time.time()
        #     dset.id.refresh()
        #     file_cap = dset.shape[0]
        #     written_batches = file_cap/batch_size 
        #     if written_batches > batch_num:
        #         break

        # This line reads the dset
        crcio = dset[batch_num*batch_size:(batch_num+1)*batch_size][:]
        cr_mbytes = crcio.nbytes/10**6
        end = time.time()
#        Optional lines for slowing down the read process to match the write
#        sleep_time = 0.0*(end-start)
#        time.sleep(sleep_time)
        cr_speed = cr_mbytes/((end-start))

        print('Read data batch %i/%i at %i MB/s' %(batch_num, written_batches, cr_speed))
        batch_num += 1



