#!/usr/bin/env python
""":py:class:`Graphics` wrapping methods for matplotlib
=======================================================

Usage::

    import psana.pyalgos.generic.Graphics as gr

    # Methods
    ddict_for_keys = dict_subset(d, keys)
    fig = gr.figure(figsize=(13,12), title='Image', dpi=80, facecolor='w', edgecolor='w', frameon=True, move=None)
    gr.move_fig(fig, x0=200, y0=100)
    gr.move(x0=200, y0=100)
    axim, axcb = gr.fig_axes(fig, windows=((0.05,  0.03, 0.87, 0.93), (0.923, 0.03, 0.02, 0.93)), **kwa)
    fig, axis = gr.add_axes(fig, axwin=(0.05, 0.03, 0.87, 0.93))
    fig, axim, axcb = gr.fig_img_cbar_axes(fig=None, win_axim=(0.05,0.03,0.87,0.93), win_axcb=(0.923,0.03,0.02,0.93))
    gr.set_win_title(fig, titwin='Image')
    gr.add_title_labels_to_axes(axes, title=None, xlabel=None, ylabel=None, fslab=14, fstit=20, color='k')
    gr.show(mode=None)
    gr.draw()
    gr.draw_fig(fig)
    gr.save_plt(fname='img.png', verb=True)
    gr.save_fig(fig, fname='img.png', verb=True)
    gr.save(fname='img.png', do_save=True, pbits=0o377)

    hi = gr.pp_hist(axis, x, **kwa) # pyplot.hist
    hi = gr.hist(axhi, arr, bins=None, amp_range=None, weights=None, color=None, log=False)
    imsh = gr.imshow(axim, img, amp_range=None, extent=None, interpolation='nearest', aspect='auto', origin='upper', orientation='horizontal', cmap='inferno')
    cbar = gr.colorbar(fig, imsh, axcb, orientation='vertical', amp_range=None)
    imsh, cbar = gr.imshow_cbar(fig, axim, axcb, img, amin=None, amax=None, extent=None, interpolation='nearest', aspect='auto', origin='upper', orientation='vertical', cmap='inferno')

    gr.drawCircle(axes, xy0, radius, linewidth=1, color='w', fill=False)
    gr.drawCenter(axes, xy0, s=10, linewidth=1, color='w')
    gr.drawLine(axes, xarr, yarr, s=10, linewidth=1, color='w')
    gr.drawRectangle(axes, xy, width, height, linewidth=1, color='w')

    # Depricated methods from pyimgalgos.GlobalGraphics.py added for compatability

    fig, axhi, hi = gr.hist1d(arr, bins=None, amp_range=None, weights=None, color=None, show_stat=True,
                              log=False, figsize=(6,5), axwin=(0.15, 0.12, 0.78, 0.80), title=None,
                              xlabel=None, ylabel=None, titwin=None)
    axim = gr.plotImageLarge(arr, img_range=None, amp_range=None, figsize=(12,10), title='Image', origin='upper',
                             window=(0.05, 0.03, 0.94, 0.94), cmap='inferno')
    fig, ax = gr.plotGraph(x,y, figsize=(5,10), window=(0.15, 0.10, 0.78, 0.86), pfmt='b-', lw=1)

See:
  - :py:class:`Utils`
  - :py:class:`NDArrUtils`
  - :py:class:`Graphics`
  - :py:class:`NDArrGenerators`
  - `matplotlib <https://matplotlib.org/contents.html>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Modified on 2018-01-25 by Mikhail Dubrovin
"""

import os
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
#export LIBGL_ALWAYS_INDIRECT=1

import logging
logger = logging.getLogger('Graphics')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import numpy as np
from time import time, localtime, strftime
from math import log10, sqrt

import matplotlib
import matplotlib.pyplot  as plt
import matplotlib.lines   as lines
import matplotlib.patches as patches

#logging.getLogger('matplotlib').setLevel(logging.WARNING)
plt.rcParams.update({'figure.max_open_warning': 0}) #get rid of warning: More than 20 figures have been opened.


def dict_subset(d, keys):
    return {k:v for k,v in d.items() if k in keys}


def figure(**kwa):
    """ Creates and returns figure.
        local pars: title='Image', move=None
    """
    # redefine default parameters for plt.figure
    kwa.setdefault('figsize',(13,12))
    kwa.setdefault('dpi',80)
    kwa.setdefault('facecolor','w')
    kwa.setdefault('edgecolor','w')
    kwa.setdefault('frameon', True)
    #kwa.setdefault('FigureClass', <class 'matplotlib.figure.Figure'>)
    kwa.setdefault('clear', False)
    kwa.setdefault('linewidth', 1)
    #kwa.setdefault('subplotpars', )
    kwa.setdefault('tight_layout', False)
    kwa.setdefault('constrained_layout', False)
    kwa_f = dict_subset(kwa, ('num', 'figsize', 'dpi', 'facecolor', 'edgecolor', 'frameon', 'FigureClass', 'clear',\
               'linewidth', 'subplotpars', 'tight_layout'))  #, 'constrained_layout'))
    fig = plt.figure(**kwa_f)
    title = kwa.get('title', 'Image')
    move  = kwa.get('move', None)
    if title: fig.canvas.manager.set_window_title(title) #, **kwa)
    if move: move_fig(fig, x0=move[0], y0=move[1])
    return fig


def set_win_title(fig, titwin='Image', **kwa):
    fig.canvas.manager.set_window_title(titwin, **kwa)


def move_fig(fig, x0=200, y0=100):
    #fig.canvas.manager.window.geometry('+%d+%d' % (x0, y0)) # in previous version of matplotlib
    backend = matplotlib.get_backend()
    #logger.debug('matplotlib.get_backend(): %s' % backend)
    if backend == 'TkAgg': # this is our case
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (x0, y0))
    elif backend == 'WXAgg':
        fig.canvas.manager.window.SetPosition((x0, y0))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        fig.canvas.manager.window.move(x0, y0)


def move(x0=200,y0=100) :
    move_fig(plt.gcf(), x0, y0)


def fig_axes(fig, **kwa):
    """ Returns list of figure axes for input list of windows
    """
    windows = kwa.get('windows', ((0.05,  0.03, 0.87, 0.93), (0.923, 0.03, 0.02, 0.93)))
    kwa.setdefault('projection',None) # projection{None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}, optional
    kwa.setdefault('polar', False)
    #kwa.setdefault('sharex', )  #sharex, shareyAxes, optional - Share the x or y axis with sharex and/or sharey.
    #kwa.setdefault('sharey',)   # The axis will have the same limits, ticks, and scale as the axis of the shared axes.
    #kwa.setdefault('label', '') # label str -A label for the returned axes.
    kwa_aa = dict_subset(kwa,\
     ('projection','polar','sharex','sharey','label',\
      'adjustable', 'agg_filter','alpha','anchor','aspect','autoscale_on','autoscalex_on','autoscaley_on',\
      'axes_locator','axisbelow','clip_box','clip_on','clip_path','contains','facecolor','fc',\
      'figure','frame_on','gid','in_layout','label','navigate','navigate_mode','path_effects',\
      'picker','position','prop_cycle','rasterization_zorder','rasterized','sketch_params',\
      'snap','title','transform','url','visible',\
      'xbound','xlabel','xlim','xmargin','xscale','xticklabels','xticklabels','xticks',\
      'ybound','ylabel','ylim','ymargin','yscale','yticklabels','yticks','zorder'))

    return [fig.add_axes(w, **kwa_aa) for w in windows]


def add_axes(fig, axwin=(0.05, 0.03, 0.87, 0.93), **kwa):
    """Add axes to figure from input list of windows.
    """
    return fig.add_axes(axwin, **kwa)


def fig_img_axes(fig=None, win_axim=(0.08, 0.05, 0.89, 0.93), **kwa):
    """ Returns figure and image axes
    """
    _fig = figure(figsize=(6,5)) if fig is None else fig
    axim = _fig.add_axes(win_axim, **kwa)
    return _fig, axim


def fig_img_cbar_axes(fig=None,\
                      win_axim=(0.05,  0.05, 0.87, 0.93),\
                      win_axcb=(0.923, 0.05, 0.02, 0.93),\
                      **kwa):
    """ Returns figure and axes for image and color bar
    """
    _fig = figure() if fig is None else fig
    return _fig,\
           _fig.add_axes(win_axim, **kwa),\
           _fig.add_axes(win_axcb, **kwa)


FYMIN, FYMAX = 0.050, 0.90
def fig_img_cbar_hist_axes(fig=None,\
                      win_axim=(0.02,  FYMIN, 0.8,  FYMAX),\
                      win_axcb=(0.915, FYMIN, 0.01, FYMAX),\
                      win_axhi=(0.76,  FYMIN, 0.15, FYMAX),\
                      **kwa):
    """ Returns figure and axes for image, color bar, and spectral histogram
    """
    _fig = figure() if fig is None else fig
    return _fig,\
           _fig.add_axes(win_axim, **kwa),\
           _fig.add_axes(win_axcb, **kwa),\
           _fig.add_axes(win_axhi, **kwa)


def add_title_labels_to_axes(axes, title=None, xlabel=None, ylabel=None, fslab=14, fstit=20, color='k', **kwa):
    if title  is not None: axes.set_title(title, color=color, fontsize=fstit, **kwa)
    if xlabel is not None: axes.set_xlabel(xlabel, fontsize=fslab, **kwa)
    if ylabel is not None: axes.set_ylabel(ylabel, fontsize=fslab, **kwa)


def show(mode=None):
    if mode is None: plt.ioff() # hold contraol at show() (connect to keyboard for controllable re-drawing)
    else           : plt.ion()  # do not hold control
    plt.pause(0.001) # hack to make it work... othervise show() does not work...
    plt.show()


def draw():
    plt.draw()


def draw_fig(fig):
    fig.canvas.draw()


def save_plt(fname='img.png', verb=True, **kwa):
    if verb: print('Save plot in file: %s' % fname)
    plt.savefig(fname, **kwa)


def save_fig(fig, fname='img.png', prefix=None, suffix='.png', verb=True, **kwa):
    path = fname
    if prefix is not None:
        ts = strftime('%Y-%m-%dT%H%M%S', localtime(time()))
        path='%s%s%s' % (prefix,ts,suffix)
    if verb: print('Save figure in file: %s' % path)
    fig.savefig(path, **kwa)


def save(fname='img.png', do_save=True, verb=True, **kwa):
    if not do_save: return
    save_plt(fname, verb, **kwa)


def proc_stat(weights, bins):
    center = np.array([0.5*(bins[i] + bins[i+1]) for i,w in enumerate(weights)])

    sum_w  = weights.sum()
    if sum_w <= 0: return  0, 0, 0, 0, 0, 0, 0, 0, 0

    sum_w2 = (weights*weights).sum()
    neff   = sum_w*sum_w/sum_w2 if sum_w2>0 else 0
    sum_1  = (weights*center).sum()
    mean = sum_1/sum_w
    d      = center - mean
    d2     = d * d
    wd2    = weights*d2
    m2     = (wd2)   .sum() / sum_w
    m3     = (wd2*d) .sum() / sum_w
    m4     = (wd2*d2).sum() / sum_w

    rms  = sqrt(m2) if m2>0 else 0
    rms2 = m2

    err_mean = rms/sqrt(neff)
    err_rms  = err_mean/sqrt(2)

    skew, kurt, var_4 = 0, 0, 0

    if rms>0 and rms2>0:
        skew  = m3/(rms2 * rms)
        kurt  = m4/(rms2 * rms2) - 3
        var_4 = (m4 - rms2*rms2*(neff-3)/(neff-1))/neff if neff>1 else 0
    err_err = sqrt(sqrt(var_4)) if var_4>0 else 0
    #print  'mean:%f, rms:%f, err_mean:%f, err_rms:%f, neff:%f' % (mean, rms, err_mean, err_rms, neff)
    #print  'skew:%f, kurt:%f, err_err:%f' % (skew, kurt, err_err)
    return mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w


def add_stat_text(axhi, weights, bins):
    mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w = proc_stat(weights,bins)
    pm = r'$\pm$'
    txt  = 'Entries=%d\nMean=%.2f%s%.2f\nRMS=%.2f%s%.2f\n' % (sum_w, mean, pm, err_mean, rms, pm, err_rms)
    txt += r'$\gamma1$=%.3f  $\gamma2$=%.3f' % (skew, kurt)
    xb,xe = axhi.get_xlim()
    yb,ye = axhi.get_ylim()
    x = xb + (xe-xb)*0.98
    y = yb + (ye-yb)*0.95

    if axhi.get_yscale() == 'log':
        log_yb, log_ye = log10(yb), log10(ye)
        log_y = log_yb + (log_ye-log_yb)*0.95
        y = 10**log_y

    axhi.text(x, y, txt, fontsize=10, color='k',
              horizontalalignment='right',
              verticalalignment='top',
              rotation=0)


def hist(axhi, arr, bins=None, amp_range=None, weights=None, color=None, log=False, **kwa):
    """Makes historgam from input array of values (arr), which are sorted in number of bins (bins) in the range (amp_range=(amin,amax))
    """
    #axhi.cla()
    hi = axhi.hist(arr.ravel(), bins=bins, range=amp_range, weights=weights, color=color, log=log, **kwa)
    ori = kwa.get('orientation', None)
    if amp_range is not None and ori is not None:
        if ori[0]=='v': axhi.set_xlim(amp_range)
        if ori[0]=='h': axhi.set_ylim(amp_range)
    wei, bins, patches = hi
    add_stat_text(axhi, wei, bins)
    return hi


def pp_hist(axis, x, **kwa):
    """ matplotlib.pyplot.hist(x,
                       bins=10,
                       range=None,
                       normed=False,
                       weights=None,
                       cumulative=False,
                       bottom=None,
                       histtype=u'bar',
                       align=u'mid',
                       orientation=u'vertical',
                       rwidth=None,
                       log=False,
                       color=None,
                       label=None,
                       stacked=False,
                       hold=None,
                       **kwargs)
    """
    return axis.hist(x, **dict_subset(kwa,\
           ('bins', 'range', 'normed', 'weights', 'cumulative', 'bottom', 'histtype', 'align',\
            'orientation', 'rwidth', 'log', 'color', 'label', 'stacked', 'hold')))


def imshow(axim, img, amp_range=None, extent=None,\
           interpolation='nearest', aspect='auto', origin='upper',\
           orientation='horizontal', cmap='inferno', **kwa):
    """
    extent - list of four image physical limits for labeling,
    cmap: 'jet', 'gray_r', 'inferno'
    #axim.cla()
    """
    imsh = axim.imshow(img, interpolation=interpolation, aspect=aspect, origin=origin, extent=extent, cmap=cmap, **kwa)
    axim.autoscale(False)
    if amp_range is not None: imsh.set_clim(amp_range[0],amp_range[1])
    return imsh


def colorbar(fig, imsh, axcb, orientation='vertical', amp_range=None, **kwa):
    """
    orientation = 'horizontal'
    amp_range = (-10,50)
    """
    if amp_range is not None: imsh.set_clim(amp_range[0],amp_range[1])
    cbar = fig.colorbar(imsh, cax=axcb, orientation=orientation, **kwa)
    return cbar


def imshow_cbar(fig, axim, axcb, img, amin=None, amax=None, **kwa):
    """
    extent - list of four image physical limits for labeling,
    cmap: 'gray_r'
    #axim.cla()
    """
    orientation = kwa.pop('orientation', 'vertical') # because imshow does not have it

    axim.cla()
    if img is None: return
    imsh = axim.imshow(img,\
           cmap=kwa.pop('cmap', 'inferno'),\
           norm=kwa.pop('norm',None),\
           aspect=kwa.pop('aspect', 'auto'),\
           interpolation=kwa.pop('interpolation', 'nearest'),\
           alpha=kwa.pop('alpha',None),\
           vmin=amin,\
           vmax=amax,\
           origin=kwa.pop('origin', 'upper'),\
           extent=kwa.pop('extent', None),\
           filternorm=kwa.pop('filternorm',True),\
           filterrad=kwa.pop('filterrad',4.0),\
           resample=kwa.pop('resample',None),\
           url=kwa.pop('url',None),\
           data=kwa.pop('data',None),\
           **kwa)
    axim.autoscale(False)
    ave = np.mean(img) if amin is None and amax is None else None
    rms = np.std(img)  if amin is None and amax is None else None
    cmin = amin if amin is not None else ave-1*rms if ave is not None else None
    cmax = amax if amax is not None else ave+3*rms if ave is not None else None
    if cmin is not None: imsh.set_clim(cmin, cmax)

    cbar = fig.colorbar(imsh, cax=axcb, orientation=orientation)
    return imsh, cbar


def fig_img_cbar(img, **kwa):
    fig = figure(figsize=kwa.pop('figsize', (12,11)))
    axim, axcb = fig_axes(fig, windows=((0.06, 0.03, 0.87, 0.93), (0.923,0.03, 0.02, 0.93)))
    imsh = axim.imshow(img, **kwa)
    imsh.set_clim(kwa.get('vmin', None), kwa.get('vmax', None))
    cbar = fig.colorbar(imsh, cax=axcb, orientation='vertical')
    return fig, axim, axcb, imsh, cbar


def fig_img_proj_cbar(img, **kwa):
    """image and its r-phi projection
    """
    fig = figure(figsize=kwa.pop('figsize', (6,12)))
    fymin, fymax = 0.050, 0.90
    winds =((0.07,  fymin, 0.685, fymax),\
            (0.76,  fymin, 0.15, fymax),\
            (0.915, fymin, 0.01, fymax))

    axim, axhi, axcb = fig_axes(fig, windows=winds)
    imsh = axim.imshow(img, **kwa)
    imsh.set_clim(kwa.get('vmin', None), kwa.get('vmax', None))
    cbar = fig.colorbar(imsh, cax=axcb, orientation='vertical')
    axim.grid(b=None, which='both', axis='both')#, **kwargs)'major'

    sh = img.shape
    w = np.sum(img, axis=1)
    phimin, phimax, radmin, radmax = kwa.get('extent', (0, 360, 1, 100))
    hbins = np.linspace(radmin, radmax, num=sh[0], endpoint=False)

    #print(info_ndarr(img,'r-phi img'))
    #print(info_ndarr(w,'r-phi weights'))
    #print(info_ndarr(hbins,'hbins'))

    kwh={'bins'       : kwa.get('bins', img.shape[0]),\
         'range'      : kwa.get('range', (radmin, radmax)),\
         'weights'    : kwa.get('weights', w),\
         'color'      : kwa.get('color', 'lightgreen'),\
         'log'        : kwa.get('log',False),\
         'bottom'     : kwa.get('bottom', 0),\
         'align'      : kwa.get('align', 'mid'),\
         'histtype'   : kwa.get('histtype',u'bar'),\
         'label'      : kwa.get('label', ''),\
         'orientation': kwa.get('orientation',u'horizontal'),\
        }

    axhi.set_ylim((radmin, radmax))
    axhi.set_yticklabels([]) # removes axes labels, not ticks
    axhi.tick_params(axis='y', direction='in')

    wei, bins, patches = his = pp_hist(axhi, hbins, **kwh)
    add_stat_text(axhi, wei, bins)

    #gr.add_title_labels_to_axes(axim, title='r vs $\phi$', xlabel='$\phi$, deg', ylabel='r, mm')#, fslab=14, fstit=20, color='k')
    #gr.draw_fig(fig)
    return fig, axim, axcb, imsh, cbar


def drawCircle(axes, xy0, radius, **kwa):
    kwa.setdefault('radius', radius)
    kwa.setdefault('linewidth', 1)
    kwa.setdefault('color', 'w')
    kwa.setdefault('fill', False)
    circ = patches.Circle(xy0, **kwa)
    axes.add_artist(circ)


def drawCenter(axes, xy0, **kwa):
    s = kwa.pop('s', 10)
    kwa.setdefault('linewidth', 1)
    kwa.setdefault('color', 'w')
    xc, yc = xy0
    d = 0.15*s
    arrx = (xc+s, xc-s, xc-d, xc,   xc)
    arry = (yc,   yc,   yc-d, yc-s, yc+s)
    line = lines.Line2D(arrx, arry, **kwa)
    axes.add_artist(line)


def drawLine(axes, xarr, yarr, **kwa):
    kwa.setdefault('linewidth', 1)
    kwa.setdefault('color', 'w')
    #print('drawLine kwa keys: %s' % str(kwa.keys()))
    kwal = dict_subset(kwa, ('color','linewidth','linestyle','marker','markersize',\
           'markeredgewidth','markeredgecolor','markerfacecolor','markerfacecoloralt','fillstyle',\
           'antialiased','dash_capstyle','solid_capstyle','dash_joinstyle','solid_joinstyle',\
           'pickradius','drawstyle','markevery'))
    line = lines.Line2D(xarr, yarr, **kwal)
    axes.add_artist(line)


def drawRectangle(axes, xy, width, height, **kwa):
    kwa.setdefault('linewidth', 1)
    kwa.setdefault('color', 'w')
    rect = patches.Rectangle(xy, width, height, **kwa)
    axes.add_artist(rect)

#--------------------------------
# DEPRICATED from GlobalGraphics
#--------------------------------

def plotImageLarge(arr, img_range=None, amp_range=None, figsize=(12,10), title='Image', origin='upper', window=(0.05,  0.03, 0.94, 0.94), cmap='inferno'):
    fig  = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    axim = fig.add_axes(window)
    imsh = axim.imshow(arr, interpolation='nearest', aspect='auto', origin=origin, extent=img_range, cmap=cmap)
    axim.autoscale(False)
    colb = fig.colorbar(imsh, pad=0.005, fraction=0.09, shrink=1, aspect=40) # orientation=1
    if amp_range is not None: imsh.set_clim(amp_range[0], amp_range[1])
    fig.canvas.manager.set_window_title(title)
    return axim


def hist1d(arr, bins=None, amp_range=None, weights=None, color=None, show_stat=True, log=False,\
           figsize=(6,5), axwin=(0.15, 0.12, 0.78, 0.80),\
           title=None, xlabel=None, ylabel=None, titwin=None):
    """Makes historgam from input array of values (arr), which are sorted in number of bins (bins) in the range (amp_range=(amin,amax))
    """
    #print 'hist1d: title=%s, size=%d' % (title, arr.size)
    if arr.size==0: return None, None, None
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    if   titwin is not None: fig.canvas.manager.set_window_title(titwin)
    elif title  is not None: fig.canvas.manager.set_window_title(title)
    axhi = fig.add_axes(axwin)
    hbins = bins if bins is not None else 100
    hi = axhi.hist(arr.ravel(), bins=hbins, range=amp_range, weights=weights, color=color, log=log) #, log=logYIsOn)
    if amp_range is not None: axhi.set_xlim(amp_range) # axhi.set_autoscale_on(False) # suppress autoscailing
    if title  is not None: axhi.set_title(title, color='k', fontsize=20)
    if xlabel is not None: axhi.set_xlabel(xlabel, fontsize=14)
    if ylabel is not None: axhi.set_ylabel(ylabel, fontsize=14)
    if show_stat:
        weights, bins, patches = hi
        add_stat_text(axhi, weights, bins)
    return fig, axhi, hi


def plotGraph(x,y, figsize=(5,10), window=(0.15, 0.10, 0.78, 0.86), pfmt='b-', lw=1):
    fig = plt.figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='w', frameon=True)
    ax = fig.add_axes(window)
    ax.plot(x, y, pfmt, linewidth=lw)
    return fig, ax


def img_from_pixel_arrays(rows, cols, weights=None, W=None, dtype=np.float32, vbase=0, mask_arr=None):
    """Returns image from rows, cols index arrays and associated weights or W.
       Methods like matplotlib imshow(img) plot 2-d image array oriented as matrix(rows,cols).
    """
    w = weights if weights is not None else W
    w = (w    if mask_arr is None else np.compress(mask_arr.ravel()>0, w)).ravel()
    r = (rows if mask_arr is None else np.compress(mask_arr.ravel()>0, rows)).ravel()
    c = (cols if mask_arr is None else np.compress(mask_arr.ravel()>0, cols)).ravel()

    if rows.size != cols.size \
    or (w is not None and r.size !=  w.size):
        msg = 'img_from_pixel_arrays(): input array sizes are different;' \
            + ' r.size=%d, c.size=%d, w.size=%d' % (rows.size, cols.size, w.size)
        logger.warning(msg)
        return img_default()

    rowsfl = r.ravel()
    colsfl = c.ravel()

    rsize = int(rowsfl.max())+1
    csize = int(colsfl.max())+1

    weight = w.ravel() if w is not None else np.ones_like(rowsfl)
    img = vbase*np.ones((rsize,csize), dtype=dtype)
    img[rowsfl,colsfl] = weight # Fill image array with data
    return img

#from psana.pscalib.geometry.GeometryAccess import img_from_pixel_arrays
getImageFromIndexArrays = img_from_pixel_arrays


if __name__ == "__main__":
    import sys
    sys.exit('Run test examples in examples/ex_Graphics.py')

# EOF

