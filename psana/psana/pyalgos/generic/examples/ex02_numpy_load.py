import os
import sys
import numpy as np
from psana.pyalgos.generic.NDArrUtils import print_ndarr

fname = sys.argv[1] if len(sys.argv)>1 else 'clb-cxix25615-cxids1-0-cspad-0-pedestals.npy'

if not os.path.exists(fname) :
    msg = 'file %s DOES NOT EXIST\nuse command: %s <file-name>.npy'%\
          (fname,sys.argv[0])
    sys.exit(msg)

a = np.load(fname)

#print(a[1,:])
print(a.shape)
print(a.dtype)
#print_ndarr(a, 'nda:')

