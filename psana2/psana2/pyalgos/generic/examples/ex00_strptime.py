from time import *

fmt='%Y-%m-%d %H:%M:%S%z'
t0_sec = time()
s0 = strftime(fmt, localtime(t0_sec))
#s0 = strftime(fmt, gmtime(t0_sec))
print('Current time stamp: %s' % s0)

s0='2008-01-01 00:00:00+0000'
print('Input time stamp: %s' % s0)

struc = strptime(s0, fmt)
print('Reconstructed time struct: ',  struc)
t1_sec = mktime(struc)
#t1_sec = 5000000000

print('Input time sec       : %d' % t1_sec)
print('Input gmtime stamp   : %s' % strftime(fmt,    gmtime(t1_sec)))
print('Input localtime stamp: %s' % strftime(fmt, localtime(t1_sec)))
