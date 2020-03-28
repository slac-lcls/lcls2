#----------
"""
    from utils import plot_peaks_on_img, data_hdf5_v0
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

class data_hdf5_v0() :
    def __init__(self, fname='/reg/g/psdm/tutorials/ami2/tmo/amox27716_run100.h5') :
        import h5py
        f=h5py.File(fname, 'r')
        self.images = f['img']
        self.nevmax = self.images.shape[0]
        self.counter=-1
        print('hdf5: %s\n  data shape: %s' % (fname, str(self.images.shape)))

    def next_image(self) :
        self.counter += 1
        if not(self.counter < self.nevmax) :
            print('WARNING counter reached its max value: %d' % self.nevmax)
            self.counter -= 1
        return self.images[self.counter,:,:]

    def print_images(self, nev_begin=0, nev_end=10) :
        for nevt in range(min(nev_begin, self.nevmax), min(nev_end, self.nevmax)) :
            print_ndarr(self.images[nevt,:,:], 'Ev:%3d'%nevt) 

#----------
