#!/usr/bin/env python
 
""" 
2020-05-01 by Mikhail DUbrovin
"""
#!/usr/bin/env python

from hashlib import sha1
import hashlib

#s = b'some very long string is here for example for the dbname'
s = b'jungfrau-170505-149520170815-3d00b0_170505-149520170815-3d00f7'

print('str:"%s" of len:%s' % (s, len(s)))
h = sha1()
h.update(s)
print('digest()', h.digest())
print('hexdigest()', h.hexdigest(), ' len:',  len(str(h.hexdigest())))
print('digest_size', h.digest_size)
print('block_size', h.block_size)
