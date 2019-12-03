#O'Grady, Paul Christopher <cpo@slac.stanford.edu>
#Wed 11/13/2019 11:42 AM
#Hi Mikhail,

#Peter?s pipico reference that I?ve been reading is here:
#https://iopscience.iop.org/article/10.1088/0953-4075/49/15/152004
#My simple ?simulation script? is below.  Thanks for thinking about it...
#chris

from psana.pyalgos.generic.NDArrUtils import np, print_ndarr
#import numpy as np

odd = np.array([1,0,1,0],dtype=float)
#odd /= np.sum(odd)
even = np.array([0,1,0,1],dtype=float)
#even /= np.sum(even)
both = odd+even
#both /= np.sum(both)

nlist = 5

all_list = [odd,even,odd,odd,even,both]*nlist
all_arr = np.vstack(all_list)
print('all_arr:\n', all_arr)


cov=None
avg=None
for i,wf in enumerate(all_arr):
    #print('proc',i,wf)
    if i==0:
        cov = np.outer(wf,wf)
        avg = wf
    else:
        cov += np.outer(wf,wf)
        avg += wf

nevts = len(all_arr)

nentries = avg.sum()
nentcov  = cov.sum()

print_ndarr('avg',avg)
print('nentcov :',nentcov)
print('nentries:',nentries)

#cov/=nevts
#cov/=len(avg)
#avg/=nevts
#avg = np.sum(all_arr,axis=0)/nevts
#bkg = np.outer(avg,avg)*nlist/nevts
#bkg = np.outer(avg,avg)/nevts
bkg = np.outer(avg,avg)/nentries

print('nevts',nevts)
print('avg:\n',avg)
print('len(avg):\n',len(avg))
print('cov:\n',cov)
print('bkg:\n',bkg)
print('cov/bkg:\n',cov/bkg)
print('cov-bkg:\n',cov-bkg)
