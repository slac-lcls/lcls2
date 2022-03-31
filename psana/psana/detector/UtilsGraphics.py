
"""
    Wrapper for graphical utils.

    import psana.detector.UtilsGraphics as ug
    from psana.detector.UtilsGraphics import *
    from psana.detector.UtilsGraphics import gr, fleximage, arr_median_limits

    img = det.raw.image(evt)
    arr = det.raw.calib(evt)
    amin, amax = arr_median_limits(arr, nneg=1, npos=3)

    flimg = fleximage(img, arr=arr, h_in=8, nneg=1, npos=3)
    flimg.update(img, arr=arr)

    gr.show(mode='DO NOT HOLD')

Created on 2020-11-09
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import psana.pyalgos.generic.Graphics as gr
#from psana.detector.NDArrUtils import info_ndarr

def arr_median_limits(arr, amin=None, amax=None, nneg=None, npos=None, fraclo=0.05, frachi=0.95):
    """ returns tuple of intensity limits (amin, amax) evaluated from arr or passed directly.
    """
    if not(None in (amin, amax)): return amin, amax

    if arr is None:
        logger.warning('arr is None, LIMITS CAN NOT BE DEFINED - RETURN 0,1')
        return 0,1
    elif None in (nneg, npos):
        qlo = np.quantile(arr, fraclo, interpolation='linear')
        qhi = np.quantile(arr, frachi, interpolation='linear')
        logger.debug('quantile(%.3f):%.1f quantile(%.3f):%.1f' % (fraclo, qlo, frachi, qhi))
        return qlo, qhi
    else:
        med = np.median(arr)
        spr = np.median(np.abs(arr-med))
        _amin, _amax = med-nneg*spr, med+npos*spr
        logger.debug('median:%.1f spread:%.1f amin:%.1f amax:%.1f' % (med, spr, _amin, _amax))
        return _amin, _amax


def gr_figure(**kwa):
    kwfig = {}
    w_in = kwa.get('w_in', 9)
    h_in = kwa.get('h_in', 8)
    return gr.plt.figure(\
        num       = kwa.get('num',None),\
        figsize   = kwa.get('figsize',(w_in, h_in)),\
        dpi       = kwa.get('dpi',80),\
        facecolor = kwa.get('facecolor','w'),\
        edgecolor = kwa.get('edgecolor','w'),\
        frameon   = kwa.get('frameon',True),\
        clear     = kwa.get('clear',False),\
        **kwfig)


def gr_imshow_cbar(fig, axim, axcb, img, **kwa):
    """imsh, cbar = gr_imshow_cbar(fig, axim, axcb, img, **kwa)"""
    kwic={'amin'         :kwa.get('amin', None),
          'amax'         :kwa.get('amax', None),
          'extent'       :kwa.get('extent', None),
          'interpolation':kwa.get('interpolation','nearest'),
          'aspect'       :kwa.get('aspect','equal'),
          'origin'       :kwa.get('origin','upper'),
          'orientation'  :kwa.get('orientation','vertical'),
          'cmap'         :kwa.get('cmap','inferno'),
          }
    return gr.imshow_cbar(fig, axim, axcb, img, **kwic)


def gr_hist(axhi, arr, **kwa):
    """his = gr_hist(axhi, arr, **kwa)
    """
    kwh={'amp_range'  : kwa.get('amp_range', (0, 1)),\
         'bins'       : kwa.get('bins', 100),\
         'weights'    : kwa.get('weights', None),\
         'color'      : kwa.get('color', None),\
         'log'        : kwa.get('log', False),\
         'bottom'     : kwa.get('bottom', None),\
         'align'      : kwa.get('align', u'mid'),\
         'histtype'   : kwa.get('histtype',u'bar'),\
         'label'      : kwa.get('label', ''),\
         'orientation': kwa.get('orientation',u'horizontal'),\
         'cumulative' : kwa.get('cumulative', False),\
         'histtype'   : kwa.get('histtype', u'bar'),\
         'rwidth'     : kwa.get('rwidth', None),\
         'stacked'    : kwa.get('stacked', False),\
         }
    return gr.hist(axhi, arr, **kwh)


class flexbase:
    def __init__(self, **kwa):
        self.amin   = kwa.get('amin', None)
        self.amax   = kwa.get('amax', None)
        self.nneg   = kwa.get('nneg', None)
        self.npos   = kwa.get('npos', None)
        self.fraclo = kwa.get('fraclo', 0.05)
        self.frachi = kwa.get('frachi', 0.95)


    def _intensity_limits(self, a, **kwa):
        """ returns tuple of intensity limits (amin, amax)
            NOTE: kwa is MUTABLE dict (NOT **kwa) because it needs (???) to be cleaned up of parameters not used in other places
        """
        arr = a if a is not None else kwa.get('arr', None)
        return arr_median_limits(arr,\
            amin   = kwa.get('amin',   self.amin),
            amax   = kwa.get('amax',   self.amax),
            nneg   = kwa.get('nneg',   self.nneg),
            npos   = kwa.get('npos',   self.npos),
            fraclo = kwa.get('fraclo', self.fraclo),
            frachi = kwa.get('frachi', self.frachi))


    def move(self, x0=100, y0=10):
        gr.move_fig(self.fig, x0, y0)


    def save(self, fname='fig.png'):
        gr.save_fig(self.fig, fname=fname, verb=True)


    def axtitle(self, title=''):
        gr.add_title_labels_to_axes(self.axim, title=title, fstit=10)
         #, xlabel=None, ylabel=None, fslab=14, fstit=20, color='k')


class fleximage(flexbase):
    def __init__(self, img, **kwa):
        flexbase.__init__(self, **kwa)
        arr = kwa.get('arr', None)
        if arr is None: arr = img #kwa['arr'] = arr = img
        amin, amax = self._intensity_limits(arr, **kwa)

        aspratio = float(img.shape[0])/float(img.shape[1]) # heigh/width

        kwa.setdefault('w_in', 9)
        kwa.setdefault('h_in', 8)
        _fig = gr_figure(**kwa)

        kwfica={}
        fymin, fymax = 0.04, 0.93
        self.fig, self.axim, self.axcb = gr.fig_img_cbar_axes(\
            fig=_fig,\
            win_axim = kwa.get('win_axim', (0.05,  fymin, 0.86, fymax)),\
            win_axcb = kwa.get('win_axcb', (0.915, fymin, 0.01, fymax)), **kwfica)

        kwa.setdefault('amin', amin)
        kwa.setdefault('amax', amax)
        self.imsh, self.cbar = gr_imshow_cbar(self.fig, self.axim, self.axcb, img, **kwa)

        gr.draw_fig(self.fig)
        #gr.show(mode=1)


    def update(self, img, **kwa):
        """use kwa: arr=arr, nneg=1, npos=3 OR arr, fraclo=0.05, frachi=0.95
        """
        amin, amax = self._intensity_limits(img, **kwa)
        self.imsh.set_data(img)
        self.imsh.set_clim(amin, amax)
        #gr.show(mode=1)


class flexhist(flexbase):
    def __init__(self, arr, **kwa):
        flexbase.__init__(self, **kwa)

        kwa.setdefault('w_in', 6)
        kwa.setdefault('h_in', 5)
        _fig = gr_figure(**kwa)

        kwfia={}
        self.fig, self.axhi = gr.fig_img_axes(\
            fig=_fig,\
            win_axim = kwa.get('win_axhi', (0.10, 0.05, 0.87, 0.90)),\
            **kwfia)

        self.update(arr, **kwa)

        gr.draw_fig(self.fig)


    def update(self, arr, **kwa):
        """use kwa: arr=arr, nneg=1, npos=3 OR arr, fraclo=0.05, frachi=0.95
        """
        amin, amax = self._intensity_limits(arr, **kwa)
        self.axhi.cla()
        kwa.setdefault('amp_range', (amin, amax))
        kwa.setdefault('orientation', u'vertical')
        if 'arr' in kwa: kwa.pop('arr')
        self.his = gr_hist(self.axhi, arr, **kwa)


    def axtitle(self, title=''):
        gr.add_title_labels_to_axes(self.axhi, title=title, fstit=10)
         #, xlabel=None, ylabel=None, fslab=14, fstit=20, color='k')


class fleximagespec(flexbase):
    def __init__(self, img, **kwa):
        """
        """
        flexbase.__init__(self, **kwa)
        #arr = kwa.setdefault('arr', img)
        arr = kwa.get('arr', None)
        if arr is None: arr = img
        amin, amax = self._intensity_limits(arr, **kwa)
        self.hcolor = kwa.get('color', 'lightgreen')
        self.hbins = kwa.get('bins', 100)

        kwa.setdefault('w_in', 14)
        kwa.setdefault('h_in', 8)
        _fig = gr_figure(**kwa)

        kwfica={}
        fymin, fymax = 0.04, 0.93
        self.fig, self.axim, self.axcb, self.axhi = gr.fig_img_cbar_hist_axes(\
            fig=_fig,\
            win_axim = kwa.get('win_axim', (0.04,  fymin, 0.71, fymax)),\
            win_axhi = kwa.get('win_axhi', (0.76,  fymin, 0.15, fymax)),\
            win_axcb = kwa.get('win_axcb', (0.915, fymin, 0.01, fymax)), **kwfica)

        kwa.setdefault('amin', amin)
        kwa.setdefault('amax', amax)
        self.imsh, self.cbar = gr_imshow_cbar(self.fig, self.axim, self.axcb, img, **kwa)

        self.update_his(arr, **kwa)
        gr.draw_fig(self.fig)


    def update_his(self, nda, **kwa):
        """use kwa: arr=arr, nneg=1, npos=3 OR arr, fraclo=0.05, frachi=0.95
        """
        amp_range = amin, amax = self._intensity_limits(nda, **kwa)

        self.axhi.cla()
        self.axhi.invert_xaxis() # anvert x-axis direction
        self.axhi.set_ylim(amp_range)
        self.axhi.set_yticklabels([]) # removes axes labels, not ticks
        self.axhi.tick_params(axis='y', direction='in')
        self.axhi.set_ylim(amp_range)
        #self.axhi.set_ylabel('V')
        #self.axhi.get_yaxis().set_visible(False) # hides axes labels and ticks

        kwa.setdefault('bins', self.hbins)
        kwa.setdefault('amp_range', amp_range)
        kwa.setdefault('color', self.hcolor)
        kwa.setdefault('orientation', u'horizontal')
        if 'arr' in kwa: kwa.pop('arr')
        self.his = gr_hist(self.axhi, nda.ravel(), **kwa)

        wei, bins, patches = self.his
        gr.add_stat_text(self.axhi, wei, bins)


    def update(self, img, **kwa):
        """
        """
        amin, amax = self._intensity_limits(img, **kwa)
        self.imsh.set_data(img)
        self.imsh.set_clim(amin, amax)
        self.axcb.set_ylim(amin, amax)

        arr = kwa.get('arr', None)
        if arr is None: arr = img
        self.update_his(arr, **kwa)

# EOF
