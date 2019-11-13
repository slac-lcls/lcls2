
#O'Grady, Paul Christopher <cpo@slac.stanford.edu>
#Wed 11/13/2019 11:42 AM
#Hi Mikhail,

#Peter?s pipico reference that I?ve been reading is here:
#https://iopscience.iop.org/article/10.1088/0953-4075/49/15/152004
#My simple ?simulation script? is below.  Thanks for thinking about it...
#chris


import numpy as np
odd = np.array([1,0,1,0],dtype=float)
odd /= np.sum(odd)
even = np.array([0,1,0,1],dtype=float)
even /= np.sum(even)
both = odd+even
both /= np.sum(both)

nlist = 10

all_list = [odd,even,both]*nlist
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

nevt = len(all_arr)
cov/=nevt
avg/=nevt
#avg = np.sum(all_arr,axis=0)/nevt
bkg = np.outer(avg,avg)*nlist/nevt
#bkg = np.outer(avg,avg)

print('nevt',nevt)
print('avg:\n',avg)
print('cov:\n',cov)
print('bkg:\n',bkg)
print('cov-bkg:\n',cov-bkg)
print('cov/bkg:\n',cov/bkg)
