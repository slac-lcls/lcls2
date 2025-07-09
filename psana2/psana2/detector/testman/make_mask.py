
from psana2.detector.NDArrUtils import info_ndarr, save_ndarray_in_textfile
import numpy as np

DTYPE_MASK = np.uint8
sh = (1024, 1024)
mask = np.ones(sh, dtype=DTYPE_MASK)
mask[100:200, 200:300] = 0

print(info_ndarr(mask, 'mask'))

fname = 'mymask.npy'
np.save(fname, mask)
print('mask saved in file %s' % fname)

fname = 'mymask.txt'
save_ndarray_in_textfile(mask, fname, 0o664, '%2d')
print('mask saved in file %s' % fname)

exit('DONE')

