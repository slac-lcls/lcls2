# Example of an HDF file reader using SWMR
# Here I read random data from a single HDF file
# The file is read from the NEH FFB (flash-based)


import h5py
import numpy as np
import time

# Prepare all the files for writing and reading

file_name = '../neh_dir/swmr/swmr_file.py'
batch_size = 10

# Function for creating random image arrays
# Some SSD controllers (SandForce) compress data before writing
# which could lead to erroneous results if the data were patterned
mb_per_img = 16
arr_size = mb_per_img*500000

def create_image(mb_per_img):
    image = np.random.randint(0,2**16,size=(250,250,4), dtype='uint16')
    arr = np.tile(image,2*mb_per_img)
    arr = arr.ravel()
    return arr


# Create the HDF file in SWMR mode
f = h5py.File(file_name, 'w', libver='latest')
dset = f.create_dataset("data", shape = (1,arr_size), chunks = (batch_size, arr_size), maxshape=(None,arr_size), dtype = 'uint16')
f.swmr_mode = True


# Main loop to write out the data
try:
    while True:
        start = time.time()
        shape = f["data"].shape
        batch_num = (shape[0]/batch_size) + 1


        f['data'].resize((shape[0]+batch_size, shape[1]))

        out_img = np.array([create_image(mb_per_img) for x in range(batch_size)])


        f['data'][-batch_size:] = out_img
        f.flush()

        end = time.time()
        cr_speed = out_img.nbytes/(10**6*(end-start))
        print('Data batch %i written at %i MB/s' % (batch_num, cr_speed))

finally:
    f.close()







