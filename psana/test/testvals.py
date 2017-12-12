import numpy as np

pgparray = np.array([[0,0,0],[0,1,2],[0,2,4]],dtype=np.float32)
testvals = {
    'array0Pgp':pgparray,
    'array1Pgp':pgparray+2,
    'floatPgp':1.0,
    'intPgp':2,
    'arrayFex':np.arange(142,148,dtype=np.float32).reshape([2,3]),
    'intFex':42,
    'floatFex':41.0,
}
