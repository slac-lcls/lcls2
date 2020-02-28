
"""module derived from psalgPeakFinder.py
"""

#import psalg_ext
from psalg_ext import peak_finder_algos
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

doPlot = int(sys.argv[1]) if len(sys.argv) > 1 else '1'

data = np.array([[  4.46530676e+00,   1.92752438e+01,   1.88353024e+01,
           2.29152584e+01,   1.43527770e+00,   2.69853268e+01,
           3.34531188e+00,  -1.46735292e-02,   2.40352535e+01,
          -4.74756539e-01],
        [  4.59452858e+01,   2.20352535e+01,   2.94953365e+01,
           2.73653316e+01,   1.75953121e+01,   1.70152340e+01,
           2.65252438e+01,   4.10652809e+01,   1.68252926e+01,
           3.29352760e+01],
        [  3.02153072e+01,   9.55297172e-01,   1.41452389e+01,
          -8.21474648e+00,   1.12652340e+01,   3.04352779e+01,
           2.47552242e+01,  -2.56472230e+00,   5.83530188e+00,
           3.18353024e+01],
        [ -1.39467835e+00,   6.83530188e+00,   1.80652828e+01,
           2.29653072e+01,  -2.22475648e+00,   6.98952408e+01,
          -7.09475183e+00,  -6.19472742e+00,   1.12952633e+01,
           3.09531188e+00],
        [ -1.15468812e+00,   1.14452877e+01,   1.40524840e+00,
          -4.54737008e-01,   1.79653072e+01,   2.46615326e+02,
           2.57353268e+01,   6.45452652e+01,   4.17453346e+01,
           3.69753151e+01],
        [  2.44853268e+01,   1.97852535e+01,   2.32453365e+01,
           2.05152340e+01,   3.73453102e+01,   1.52529529e+03,
           2.55025238e+02,   5.91252174e+01,   2.12752438e+01,
           5.46252174e+01],
        [  7.03525305e+00,   4.55652809e+01,   2.75552731e+01,
          -1.21474671e+00,   1.34065277e+02,   1.72205292e+02,
           1.21555275e+02,   1.34495331e+02,   2.49452877e+01,
          -2.95473695e+00],
        [ -1.02468324e+00,   1.89523864e+00,  -4.47028279e-02,
          -5.81472254e+00,   1.43527770e+00,   1.73553219e+01,
           1.27524352e+00,   4.16053200e+01,   4.52524328e+00,
           1.18653316e+01],
        [ -4.14697945e-01,  -9.14697945e-01,  -3.85476136e+00,
           3.85532165e+00,   2.96530700e+00,   2.83252926e+01,
           2.29753170e+01,   1.38522887e+00,   2.54526305e+00,
          -1.01467347e+00]], dtype=np.float32)
mask = np.ones_like(data, dtype=np.uint16)

# step 1
pfalgos = peak_finder_algos(pbits=0)#, lim_peaks=2048)

# step 2
pfalgos.set_peak_selection_parameters(npix_min=2, npix_max=100, amax_thr=200, atot_thr=600, son_min=7)

peaks = pfalgos.peak_finder_v4r3_d2(data, mask, thr_low=1, thr_high=10, rank=5, r0=4, dr=2)
print('XXX pfalgos.peak_finder_v4r3_d2:', peaks)

if len(peaks)>0 : print('dir(peaks[0]):', dir(peaks[0]))
for p in peaks :
    #print('  pfalgos:', p.parameters())
    print('  row:%4d, col:%4d, npix:%4d, son:%4.1f amp_tot:%4.1f' % (p.row, p.col, p.npix, p.son, p.amp_tot))

#====================
sys.exit('TEST EXIT')
#====================

# step 3
rows, cols, intens = \
pfalgos.peak_finder_v3r3_d2(data, mask, rank=3, r0=4, dr=2, nsigm=0)
print("Peaks found (rows, cols, intens): ", rows, cols, intens)
assert(len(rows)==1)
assert(intens[0]>1500)

rows1, cols1, intens1 = \
pfalgos.peak_finder_v3r3_d2(data+6, mask, rank=3, r0=4, dr=2, nsigm=0)

print("intens:",intens)
print("intens1:",intens1)
temp = intens
temp[0]=999
print("intens:",intens)
print("intens1:",intens1)

if doPlot:
    fig, ax = plt.subplots()
    ax.imshow(data, interpolation='none')
    ax.scatter(cols, rows, s=50, facecolors='none', edgecolors='r')
    plt.show()

tr = rows
tr[0] = 888

for i in range(10):
    rows, cols, intens = \
        pfalgos.peak_finder_v3r3_d2(data, mask, rank=3, r0=4, dr=2, nsigm=0)
    print("Peaks found (rows, cols, intens): ", rows, cols, intens)
print("intens:",intens)
print("temp:",temp)
print("tr:",tr)
