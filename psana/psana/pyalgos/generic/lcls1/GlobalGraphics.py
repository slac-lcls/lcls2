#!@PYTHON@
####!/usr/bin/env python
#------------------------------
"""
:py:class:`GlobalGraphics` - collection of global graphical methods
===================================================================

Usage::

    import pyimgalgos.GlobalGraphics as gg

    # Methods

    fig, axim, axcb = gg.fig_axes(figsize=(13,12), title='Image', dpi=80, win_axim=(0.05, 0.03, 0.87, 0.93), win_axcb=(0.923, 0.03, 0.02, 0.93))
    fig, axim, axcb, imsh = gg.fig_axim_axcb_imsh(figsize=(13,12), title='Image', dpi=80, win_axim=(0.05, 0.03, 0.87, 0.93),
                            win_axcb=(0.923, 0.03, 0.02, 0.93), arr2d=np.zeros((10,10)), origin='upper')
    gg.plot_imgcb(fig, axim, axcb, imsh, arr2d, amin=None, amax=None, origin='upper', title=None, cmap='inferno')
    gg.plot_img(img, mode=None, amin=None, amax=None, cmap='inferno')
    gg.plot_peaks_on_img(peaks, axim, iX, iY, color='w', pbits=0, lw=2)
    size = gg.size_of_shape(shape=(2,3,8))
    arr = gg.getArrangedImage(shape=(40,60))
    arr = gg.getRandomImage(mu=200, sigma=25, shape=(40,60))
    hi = gg.getImageAs2DHist(iX,iY,W=None)
    img = gg.getImageFromIndexArrays(iX,iY,W=None)
    fig, axhi, hi = gg.plotHistogram(arr, amp_range=None, figsize=(6,6), bins=None, title='', window=(0.15, 0.10, 0.78, 0.82))
    fig, axhi, hi = gg.hist1d(arr, bins=None, amp_range=None, weights=None, color=None, show_stat=True,
                              log=False, figsize=(6,5), axwin=(0.15, 0.12, 0.78, 0.80), title=None, 
                              xlabel=None, ylabel=None, titwin=None)
    axim = gg.plotImageLarge(arr, img_range=None, amp_range=None, figsize=(12,10), title='Image', origin='upper', 
                             window=(0.05, 0.03, 0.94, 0.94), cmap='inferno')
    gg.plotImageAndSpectrum(arr, amp_range=None, cmap='inferno') 
    fig, ax = gg.plotGraph(x,y, figsize=(5,10), window=(0.15, 0.10, 0.78, 0.86), pfmt='b-', lw=1)
    gg.drawCircle(axes, xy0, radius, linewidth=1, color='w', fill=False)
    gg.drawCircle(axes, xy0, radius, linewidth=1, color='w', fill=False)
    gg.drawCenter(axes, xy0, s=10, linewidth=1, color='w')
    gg.drawLine(axes, xarr, yarr, s=10, linewidth=1, color='w')
    gg.drawRectangle(axes, xy, width, height, linewidth=1, color='w')
    gg.save(fname='img.png', do_save=True, pbits=0377)
    gg.save_fig(fig, fname='img.png', do_save=True, pbits=0377)
    gg.move(x0=200,y0=100)
    gg.move_fig(fig, x0=200, y0=100)
    gg.show(mode=None)

See:
  - :py:class:`Graphics`
  - :py:class:`GlobalGraphics`
  - :py:class:`NDArrGenerators`
  - `matplotlib <https://matplotlib.org/contents.html>`_.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

Created in 2015 by Mikhail Dubrovin
"""
#--------------------------------

import sys
import numpy as np

import matplotlib
#if matplotlib.get_backend() != 'Qt4Agg' : matplotlib.use('Qt4Agg')

import matplotlib.pyplot  as plt
import matplotlib.lines   as lines
import matplotlib.patches as patches

from CalibManager.PlotImgSpeWidget import add_stat_text

#------------------------------

class Storage :
    def __init__(self) :
        pass

#------------------------------
store = Storage() # singleton
#------------------------------

def fig_axes(figsize=(13,12), title='Image', dpi=80, \
             win_axim=(0.05,  0.03, 0.87, 0.93), \
             win_axcb=(0.923, 0.03, 0.02, 0.93)) :
    """ Creates and returns figure, and axes for image and color bar
    """
    fig  = plt.figure(figsize=figsize, dpi=dpi, facecolor='w', edgecolor='w', frameon=True)
    axim = fig.add_axes(win_axim)
    axcb = fig.add_axes(win_axcb)
    fig.canvas.set_window_title(title)
    store.fig, store.axim, store.axcb = fig, axim, axcb
    return fig, axim, axcb

#------------------------------

def fig_axim_axcb_imsh(figsize=(13,12), title='Image', dpi=80,\
                       win_axim=(0.05,  0.03, 0.87, 0.93),\
                       win_axcb=(0.923, 0.03, 0.02, 0.93),\
                       arr2d=np.zeros((10,10)), origin='upper') :
    """ Creates and returns figure, axes for image and color bar, imshow object
    """
    fig  = plt.figure(figsize=figsize, dpi=dpi, facecolor='w', edgecolor='w', frameon=True)
    axim = fig.add_axes(win_axim)
    axcb = fig.add_axes(win_axcb)
    fig.canvas.set_window_title(title)
    imsh = axim.imshow(arr2d, interpolation='nearest', aspect='auto', origin=origin) 
    return fig, axim, axcb, imsh

#------------------------------

def plot_imgcb(fig, axim, axcb, imsh, arr2d, amin=None, amax=None, origin='upper', title=None, cmap='inferno') :
    if arr2d is None : return
    if imsh is not None : imsh.set_data(arr2d)
    else : imsh = axim.imshow(arr2d, interpolation='nearest', aspect='auto', origin=origin, cmap=cmap) 
    ave = np.mean(arr2d) if amin is None and amax is None else None
    rms = np.std(arr2d)  if amin is None and amax is None else None
    #print 'img ave = %s, rms = %s' % (str(ave), str(rms))
    cmin = amin if amin is not None else ave-1*rms if ave is not None else None
    cmax = amax if amax is not None else ave+3*rms if ave is not None else None
    if cmin is not None : imsh.set_clim(cmin, cmax)
    colb = fig.colorbar(imsh, cax=axcb) # , orientation='horizontal')
    if title is not None : fig.canvas.set_window_title(title)    

#------------------------------

def plot_img(img, mode=None, amin=None, amax=None, cmap='inferno') :
    
    fig, axim, axcb = store.fig, store.axim, store.axcb

    axim.cla()
    imsh = axim.imshow(img, interpolation='nearest', aspect='auto', origin='upper', cmap=cmap) # extent=img_range)
    colb = fig.colorbar(imsh, cax=axcb) # , orientation='horizontal')

    ave = np.mean(img) if amin is not None or amax is not None else None
    rms = np.std(img)  if amin is not None or amax is not None else None
    #print 'img ave = %f, rms = %f' % (ave, rms)
    store.amin = amin if amin is not None else ave-1*rms
    store.amax = amax if amax is not None else ave+5*rms

    imsh.set_clim(store.amin, store.amax)

    #print_help(1)

    if mode is None : plt.ioff() # hold contraol at show() (connect to keyboard for controllable re-drawing)
    else            : plt.ion()  # do not hold control

    #fig.canvas.draw()
    plt.show()

##-----------------------------

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
    for rec in peaks :
        s, r, c, amax, atot, npix = rec[0:6]
        if pbits & 1 : print 's, r, c, amax, atot, npix=', s, r, c, amax, atot, npix
        inds = (int(s),int(r),int(c)) if iX.ndim > 2 else (int(r),int(c))
        x = iX[inds]
        y = iY[inds]
        if pbits & 2 : print ' x,y=',x,y        
        xyc = (y,x)
        #r0  = 2+3*atot/anorm
        r0  = 5
        circ = patches.Circle(xyc, radius=r0, linewidth=lw, color=color, fill=False)
        axim.add_artist(circ)

##-----------------------------

def size_of_shape(shape=(2,3,8)) :
    size = 1
    for d in shape : size *= d
    return size

#------------------------------

def getArrangedImage(shape=(40,60)) :
    arr = np.arange(size_of_shape(shape))
    arr.shape = shape
    return arr

#--------------------------------

def getRandomImage(mu=200, sigma=25, shape=(40,60)) :
    arr = mu + sigma*np.random.standard_normal(shape)
    return arr

#------------------------------

def getImageAs2DHist(iX,iY,W=None) :
    """Makes image from iX, iY coordinate index arrays and associated weights, using np.histogram2d(...).
    """
    xsize = np.ceil(iX).max()+1 
    ysize = np.ceil(iY).max()+1
    if W is None : weights = None
    else         : weights = W.flatten()
    H,Xedges,Yedges = np.histogram2d(iX.flatten(), iY.flatten(), bins=(xsize,ysize), range=((-0.5,xsize-0.5),(-0.5,ysize-0.5)), normed=False, weights=weights) 
    return H

#------------------------------

def getImageFromIndexArrays(iX,iY,W=None) :
    """Makes image from iX, iY coordinate index arrays and associated weights, using indexed array.
    """
    xsize = iX.max()+1 
    ysize = iY.max()+1
    if W is None : weight = np.ones_like(iX)
    else         : weight = W
    img = np.zeros((xsize,ysize), dtype=np.float32)
    img[iX,iY] = weight # Fill image array with data 
    return img

#--------------------------------

def plotHistogram(arr, amp_range=None, figsize=(6,6), bins=None, title='', window=(0.15, 0.10, 0.78, 0.82)) :
    """Makes historgam from input array of values (arr), which are sorted in number of bins (bins) in the range (amp_range=(amin,amax))
    """
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w',edgecolor='w', frameon=True)
    axhi = fig.add_axes(window)
    hbins = bins if bins is not None else 100
    hi = axhi.hist(arr.flatten(), bins=hbins, range=amp_range) #, log=logYIsOn)
    #fig.canvas.set_window_title(title)
    axhi.set_title(title, color='k', fontsize=20)
    return fig, axhi, hi

#--------------------------------

def hist1d(arr, bins=None, amp_range=None, weights=None, color=None, show_stat=True, log=False,\
           figsize=(6,5), axwin=(0.15, 0.12, 0.78, 0.80),\
           title=None, xlabel=None, ylabel=None, titwin=None) :
    """Makes historgam from input array of values (arr), which are sorted in number of bins (bins) in the range (amp_range=(amin,amax))
    """
    #print 'hist1d: title=%s, size=%d' % (title, arr.size)
    if arr.size == 0 : return None, None, None
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    if   titwin is not None : fig.canvas.set_window_title(titwin)
    elif title  is not None : fig.canvas.set_window_title(title)
    axhi = fig.add_axes(axwin)
    hbins = bins if bins is not None else 100
    hi = axhi.hist(arr.flatten(), bins=hbins, range=amp_range, weights=weights, color=color, log=log) #, log=logYIsOn)
    if amp_range is not None : axhi.set_xlim(amp_range) # axhi.set_autoscale_on(False) # suppress autoscailing
    if title  is not None : axhi.set_title(title, color='k', fontsize=20)
    if xlabel is not None : axhi.set_xlabel(xlabel, fontsize=14)
    if ylabel is not None : axhi.set_ylabel(ylabel, fontsize=14)
    if show_stat :
        weights, bins, patches = hi
        add_stat_text(axhi, weights, bins)
    return fig, axhi, hi

#--------------------------------

def plotSpectrum(arr, amp_range=None, figsize=(6,6)) : # range=(0,500)
    plotHistogram(arr, amp_range, figsize)

#--------------------------------

def plotImage(arr, img_range=None, amp_range=None, figsize=(12,5), title='Image', origin='upper', window=(0.05,  0.05, 0.95, 0.92), cmap='jet') : 
    fig  = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    axim = fig.add_axes(window)
    imsh = plt.imshow(arr, interpolation='nearest', aspect='auto', origin=origin, extent=img_range, cmap=cmap) #,extent=self.XYRange, origin='lower'
    colb = fig.colorbar(imsh, pad=0.005, fraction=0.1, shrink=1, aspect=20)
    if amp_range is not None : imsh.set_clim(amp_range[0],amp_range[1])
    #axim.set_title(title, color='b', fontsize=20)
    fig.canvas.set_window_title(title)

#--------------------------------

def plotImageAndSpectrum(arr, amp_range=None, cmap='inferno') : #range=(0,500)
    fig  = plt.figure(figsize=(15,5), dpi=80, facecolor='w', edgecolor='w', frameon=True)
    fig.canvas.set_window_title('Image And Spectrum ' + u'\u03C6')

    ax1   = plt.subplot2grid((10,10), (0,4), rowspan=10, colspan=6)
    axim1 = ax1.imshow(arr, interpolation='nearest', aspect='auto', cmap=cmap) # , origin='lower' 
    colb1 = fig.colorbar(axim1, pad=0.01, fraction=0.1, shrink=1.00, aspect=20)
    if amp_range is not None : axim1.set_clim(amp_range[0], amp_range[1])
    plt.title('Image', color='b', fontsize=20)

    ax2   = plt.subplot2grid((10,10), (0,0), rowspan=10, colspan=4)
    ax2.hist(arr.flatten(), bins=100, range=amp_range)
    plt.title('Spectrum', color='r',fontsize=20)
    plt.xlabel('Bins')
    plt.ylabel('Stat') #u'\u03C6'
    #plt.ion()

#------------------------------

def plotGraph(x,y, figsize=(5,10), window=(0.15, 0.10, 0.78, 0.86), pfmt='b-', lw=1) : 
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    ax = fig.add_axes(window)
    ax.plot(x, y, pfmt, linewidth=lw)
    return fig, ax

#------------------------------

def drawCircle(axes, xy0, radius, linewidth=1, color='w', fill=False) : 
    circ = patches.Circle(xy0, radius=radius, linewidth=linewidth, color=color, fill=fill)
    axes.add_artist(circ)


def drawCenter(axes, xy0, s=10, linewidth=1, color='w') : 
    xc, yc = xy0
    d = 0.15*s
    arrx = (xc+s, xc-s, xc-d, xc,   xc)
    arry = (yc,   yc,   yc-d, yc-s, yc+s)
    line = lines.Line2D(arrx, arry, linewidth=linewidth, color=color)   
    axes.add_artist(line)


def drawLine(axes, xarr, yarr, s=10, linewidth=1, color='w') : 
    line = lines.Line2D(xarr, yarr, linewidth=linewidth, color=color)   
    axes.add_artist(line)


def drawRectangle(axes, xy, width, height, linewidth=1, color='w') :
    rect = patches.Rectangle(xy, width, height, linewidth=linewidth, color=color)
    axes.add_artist(rect)

#------------------------------

def save(fname='img.png', do_save=True, pbits=0377) :
    if not do_save : return
    if pbits & 1 : print 'Save plot in file: %s' % fname 
    plt.savefig(fname)

#--------------------------------

def savefig(fname='img.png', do_print=True) :
    save(fname, do_save=True, pbits=0377 if do_print else 0)

#------------------------------

def save_fig(fig, fname='img.png', do_save=True, pbits=0377) :
    if not do_save : return
    if pbits & 1 : print 'Save plot in file: %s' % fname 
    fig.savefig(fname)

#--------------------------------

def move(x0=200,y0=100) :
    plt.get_current_fig_manager().window.move(x0, y0)

#--------------------------------

def move_fig(fig, x0=200, y0=100) :
    fig.canvas.manager.window.move(x0, y0)

#--------------------------------

def show(mode=None) :
    if mode is None : plt.ioff() # hold contraol at show() (connect to keyboard for controllable re-drawing)
    else            : plt.ion()  # do not hold control
    plt.pause(0.0001) # hack to make it work... othervise show() does not work...
    plt.show()

#----------------------------------------------

def main() :

    arr = getRandomImage()
    if len(sys.argv) == 1   :
        print 'Use command > python %s <test-number [1-5]>' % sys.argv[0]
        sys.exit ('Add <test-number> in command line...')

    elif sys.argv[1] == '1' : plotImage(arr, amp_range=(100,300))
    elif sys.argv[1] == '2' : plotImageAndSpectrum(arr, amp_range=(100,300))
    elif sys.argv[1] == '3' : plotHistogram(arr, amp_range=(100,300), figsize=(10,5))
    elif sys.argv[1] == '4' : plotImage(arr, amp_range=(100,300), figsize=(10,10))
    elif sys.argv[1] == '5' : plotImageLarge(arr, amp_range=(100,300), figsize=(10,10))
    elif sys.argv[1] == '6' : plotImageLarge(getArrangedImage(shape=(40,60)), figsize=(10,10))
    else :
        print 'Non-expected arguments: sys.argv=', sys.argv
        sys.exit ('Check input parameters')

    move(500,10)
    show()

#--------------------------------

if __name__ == "__main__" :

    main()
    sys.exit ( 'End of test.' )

#----------------------------------------------
