import pickle
import numpy as np

d = {1:'a', 'b':2, 3:np.array((1,2,3))}
s = pickle.dumps(d)
o = pickle.loads(s)

print('original dict:',   d)
print('serialized dict:', s)
print('restored dict:',   o)
