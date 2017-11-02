import numpy as np

pgparray = np.array([[0,0,0],[0,1,2],[0,2,4]],dtype=np.float32)
testvals = {
    'array0_pgp':pgparray,
    'array1_pgp':pgparray+2,
    'float_pgp':1.0,
    'int_pgp':2,
    'array_fex':np.arange(142,148,dtype=np.float32).reshape([2,3]),
    'int_fex':42,
    'float_fex':41.0,
}
