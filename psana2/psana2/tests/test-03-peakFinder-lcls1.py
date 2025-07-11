import peakFinder
import numpy as np
import matplotlib.pyplot as plt

calib = np.load("cxitut13_r10_32.npy")
data = calib[0]
mask = np.ones_like(data, dtype=np.uint16)
print(mask)
# step 1
pk = peakFinder.peak_finder_algos(pbits=0, lim_peaks=2048)
print("Done step1")

# step 2
pk.set_peak_selection_parameters(npix_min=2, npix_max=30, amax_thr=200, atot_thr=600, son_min=7)
print("Done step2")

# step 3
rows, cols, intens = \
pk.peak_finder_v3r3_d2(data, mask, rank=3, r0=4, dr=2, nsigm=0)
# row, col, *_ = a.peak_finder_v3r3_d2(data, mask, rank=3, r0=4, dr=2, nsigm=0)
print("Done step3")

fig, ax = plt.subplots()
ax.imshow(data, interpolation='none')
ax.scatter(cols, rows, s=50, facecolors='none', edgecolors='r')
plt.show()

print("Done plot")

data1 = np.flipud(data)
np.save("cxitut13_r10_32_flipud.npy", data1)
data1 = np.load("cxitut13_r10_32_flipud.npy")
rows1, cols1, intens1 = \
pk.peak_finder_v3r3_d2(data1, mask, rank=3, r0=4, dr=2, nsigm=0)
print("Done step3")

fig, ax = plt.subplots()
ax.imshow(data1, interpolation='none')
ax.scatter(cols1, rows1, s=50, facecolors='none', edgecolors='r')
plt.show()

print("rows1: ", rows1)
print("rows: ", rows)
