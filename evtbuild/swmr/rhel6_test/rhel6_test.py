# Script to test SWMR in the /teststripe directory
# Requires HDF 1.10+ and h5py 2.7+, or the "snake_rhel6" environment
# Call with mpirun -q -n 2 python rhel6_test.py

from mpi4py import MPI
import time
import h5py


world_comm = MPI.COMM_WORLD
rank = world_comm.Get_rank()
world_size = world_comm.Get_size()


# The first path to teststripe has the refresh problem.

file_name = '/teststripe/eliseo/rhel_swmr_test.hdf'

# The script behaves as normal with a local path.
#file_name = 'rhel_swmr_test.hdf'
   


if rank == 0:
    #Writer
    loop_file  = h5py.File(file_name, 'w', libver = 'latest')
    timestamps = loop_file.create_dataset('timestamps', shape = (1,), chunks=(1,), maxshape=(None,), dtype='i')
        
    loop_file.swmr_mode = True
    loop_file.flush()

    ts_dset = loop_file['timestamps']

    ind = 0 
    while True:
        ts_dset[-1] = ind
        ind += 1
        ts_dset.resize((ts_dset.shape[0]+1,))
        loop_file.flush()
        if ind%500 == 0:
            print('Writer says the shape is %i' % ind)

    
elif rank == 1:
    #Reader
    time.sleep(1)

 
    loop_file  = h5py.File(file_name, 'r', libver = 'latest', swmr=True)
    
    ts_dset = loop_file['timestamps']

    new_shape =0
    old_shape = 0
    while True:
        new_shape = ts_dset.shape[0]
        ts_dset[-(new_shape-old_shape)]
        time.sleep(0.001)
#        print('Reader refresh')
        ts_dset.id.refresh()
#        print('Reader refresh done')
        if new_shape%1 == 0 and new_shape != old_shape:
            old_shape = new_shape
            print('Reader says the shape is %i' % (new_shape))
else:
    pass
 #   read_files(comm, 0) # copy

