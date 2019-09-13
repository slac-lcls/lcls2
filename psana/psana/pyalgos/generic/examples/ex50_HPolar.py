import numpy as np
from psana.pyalgos.generic.HPolar import HPolar
from psana.pyalgos.generic.NDArrUtils import print_ndarr

nx = 3
ny = 2
N = 5
ystart = 1

lx = np.linspace(-nx, nx, 2*nx*N)
ly = np.linspace(ystart, ny+1, ny*N)
ly = np.concatenate((-ly[::-1], ly))

xarr, yarr = np.meshgrid(lx, ly, indexing='ij')

val = np.ones_like(xarr)

print_ndarr(lx, name='lx', first=0, last=5)
print_ndarr(ly, name='ly', first=0, last=5)
print_ndarr(xarr, name='xarr', first=0, last=5)
print_ndarr(yarr, name='yarr', first=0, last=5)


hpolar = HPolar(xarr, yarr, nradbins=5, nphibins=64)
arr_rphi = hpolar.bin_avrg_rad_phi(val)



import psana.pyalgos.generic.Graphics as gg

img = arr_rphi # yarr
gg.plotImageLarge(img, amp_range=None, figsize=(14,12), title='test-Hpolar', cmap='inferno') #'jet'
gg.show()   
