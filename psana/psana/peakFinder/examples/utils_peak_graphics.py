#----------
"""
from utils_peak_graphics import plot_peaks_on_img
"""

import matplotlib.patches as patches 

def plot_peaks_on_img(peaks, axim, iX, iY, color='w', pbits=0, lw=2) :  
    """ Draws peaks on the top of image axes (axim)
        Plots peaks from array as circles in coordinates of image.
        - peaks - 2-d list/tuple of peaks; first 6 values in each peak record should be (s, r, c, amax, atot, npix)  
        - axim - image axes
        - iX - array of x-coordinate indexes for all pixels addressed as [s, r, c] - segment, row, column
        - iX - array of y-coordinate indexes for all pixels addressed as [s, r, c] - segment, row, column
        - color - peak-ring color
        - pbits - verbosity; print 0 - nothing, +1 - peak parameters, +2 - x, y peak coordinate indexes
    """

    if peaks is None : return

    #anorm = np.average(peaks,axis=0)[4] if len(peaks)>1 else peaks[0][4] if peaks.size>0 else 100    
    for p in peaks :
        #s, r, c, amax, atot, npix = p[0:6]
        s, r, c, amax, atot, npix = 0, p.row, p.col, p.amp_max, p.amp_tot, p.npix

        if pbits & 1 : print('s, r, c, amax, atot, npix=', s, r, c, amax, atot, npix)
        inds = (int(s),int(r),int(c)) if iX.ndim>2 else (int(r),int(c))
        x=iX[inds]
        y=iY[inds]
        if pbits & 2 : print(' x,y=',x,y)
        xyc = (y,x)
        #r0  = 2+3*atot/anorm
        r0  = 5
        circ = patches.Circle(xyc, radius=r0, linewidth=lw, color=color, fill=False)
        axim.add_artist(circ)

#----------
