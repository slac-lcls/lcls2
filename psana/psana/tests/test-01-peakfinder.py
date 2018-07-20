import peakFinder as algos
import numpy as np

import psana.pyalgos.generic.Graphics as gr
from psana.pyalgos.generic.NDArrUtils import print_ndarr

doPlot = 0

#------------------------------

def plot_image(data) :
    fig, axim, axcb = gr.fig_img_cbar_axes(gr.figure(figsize=(10,5), dpi=80))
    ave, rms = np.mean(data), np.std(data)
    _,_ = gr.imshow_cbar(fig, axim, axcb, data, amin=ave-1*rms, amax=ave+5*rms, extent=None, cmap='inferno')
    return axim

#------------------------------

calib = np.load('/reg/g/psdm/detector/data_test/arrays/cxitut13_r10_32_cspad.npy')
print_ndarr(calib, 'calib')
data = calib[0]
print_ndarr(data, 'data')

mask = np.ones_like(data, dtype=np.uint16)
print(mask)
# step 1
#pk = algos.peak_finder_algos(pbits=0, lim_peaks=2048)
pk = algos.peak_finder_algos(pbits=0)
print("Done step1")

# step 2
pk.set_peak_selection_parameters(npix_min=2, npix_max=30, amax_thr=200, atot_thr=600, son_min=7)
print("Done step2")

# step 3
rows, cols, intens = \
pk.peak_finder_v3r3_d2(data, mask, rank=3, r0=4, dr=2, nsigm=0)
# row, col, *_ = a.peak_finder_v3r3_d2(data, mask, rank=3, r0=4, dr=2, nsigm=0)
print("Done step3")

#arrmax = pk.local_minima()
#print_ndarr(arrmax, 'arrmax')
#aximax = plot_image(arrmax)
#gr.move(x0=500, y0=0)
#gr.show('do_not_hold')

if doPlot:
    axim = plot_image(data)
    axim.scatter(cols, rows, s=100, facecolors='none', edgecolors='w')
    gr.show('do not hold')

#pk1 = algos.peak_finder_algos(pbits=0)
#pk1.set_peak_selection_parameters(npix_min=2, npix_max=30, amax_thr=200, atot_thr=600, son_min=7)

data1 = np.array(np.flipud(data))
#np.save("cxitut13_r10_32_flipud.npy", data1)
#data1 = np.load("cxitut13_r10_32_flipud.npy")
rows1, cols1, intens1 = \
pk.peak_finder_v3r3_d2(data1, mask, rank=3, r0=4, dr=2, nsigm=0)
print("Done step3")

#fig, ax = plt.subplots()
#ax.imshow(data1, interpolation='none')

if doPlot:
    axim1 = plot_image(data1)
    axim1.scatter(cols1, rows1, s=100, facecolors='none', edgecolors='w')
    gr.move(x0=0, y0=500)
    gr.show()

print("rows: ", len(rows), rows)
print("rows1: ", len(rows1), (184-rows1)[-1::-1])

#------------------------------
