import numpy as np

ind_max = 10 
print('expect %d indices, but array do no contain top indexes:' % ind_max)
a = (0,0,0,1,2,5,7,2,5,5,5,5,7,7,7,7,7,7)
print('input array:', a, ' len:', len(a))

b = np.bincount(a)
print('np.bincount w/o minlength :', b, ' len:', len(b))

c = np.bincount(a, weights=None, minlength=ind_max)
print('np.bincount with minlength:', c, ' len:', len(c))

#xarr, yarr = np.meshgrid(lx, ly, indexing='ij')
#val = np.ones_like(xarr)

#from psana.pyalgos.generic.HPolar import HPolar

#from psana.pyalgos.generic.NDArrUtils import print_ndarr
#print_ndarr(lx, name='lx', first=0, last=5)

#import psana.pyalgos.generic.Graphics as gg
#gg.plotImageLarge(img, amp_range=None, figsize=(14,12), title='test-Hpolar', cmap='inferno') #'jet'
#gg.show()   
