import numpy as np
import dgramCreate as dc

DETNAME = b'xpphsd'
DETTYPE = b'hsd'
DETID = b'detnum1234'
NAMESID = 0
NUM_ELEM = 2

def generate_block(det_name, det_type, det_id, names_id, num_elem):

    types = ['uint8', 'uint16', 'uint32', 'uint64',\
            'int8', 'int16', 'int32', 'int64', 'float', 'double']

    np_types = list(map(lambda x: np.dtype(x), types))

    dat_arr = []
    for i in range(num_elem):
        dat_arr_shape = tuple([np.random.randint(1, 10) for x in range(np.random.randint(1, 5))])
        arr_dtype = np.random.choice(np_types)
        if arr_dtype in (np.float, np.double):
            d_arr = np.random.random(dat_arr_shape).astype(arr_dtype)
        else:
            d_arr = np.random.randint(np.iinfo(arr_dtype).max, size=dat_arr_shape, dtype=arr_dtype)
        dat_arr.append(d_arr)

    data_block = [[dc.nameinfo(det_name, det_type, det_id, names_id)]]

    for i in range(num_elem):
        data_block.append([[b"name%i" % i, dc.alg(b"raw", [0, 0, 0])], dat_arr[i]])
    return data_block, dat_arr

VERBOSE = 1
DATA_BLOCK, DAT_ARR = generate_block(DETNAME, DETTYPE, DETID, NAMESID, NUM_ELEM)
dc.blockcreate(DATA_BLOCK)


# Read the xtc back in
with open('data.xtc', 'rb') as f:
    DATA = f.read()

# Print out the arrays, if asked.
if VERBOSE:
    DATA_BYTES = sum(map(lambda x: x.nbytes, DAT_ARR))
    D_OFFSET = len(DATA)-DATA_BYTES
    print("Data array is: \n")
    for i in range(NUM_ELEM):
        DATA_M = np.resize(np.frombuffer(DATA[D_OFFSET:D_OFFSET+DAT_ARR[i].nbytes], dtype=DAT_ARR[i].dtype), DAT_ARR[i].shape)
        D_OFFSET += DAT_ARR[i].nbytes
        print("Array %i:" % i)
        print('%s of %s' % ('x'.join(map(str, DAT_ARR[i].shape)), DATA_M.dtype.name))
        print(DATA_M, '\n')
