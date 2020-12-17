"""
    Wrapper for graphical utils.

    from psana.psana.detector.UtilsGraphics import *
    from psana.psana.detector.UtilsGraphics import gr, fleximage, arr_median_limits

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

#----

def arr_median_limits(arr, nneg=1, npos=3):
    """ returns tuple of intensity limits (amin, amax) evaluated from arr.
    """
    med = np.median(arr)
    spr = np.median(np.abs(arr-med))
    amin, amax = med-nneg*spr, med+npos*spr
    logger.debug('median:%.1f spread:%.1f amin:%.1f amax:%.1f' % (med, spr, amin, amax))
    return amin, amax


class fleximage:
    def __init__(self, img, **kwa):
        """
        """
        # set local parameters:
        arr          = kwa.setdefault('arr', img)
        self.nneg    = kwa.setdefault('nneg', 1)
        self.npos    = kwa.setdefault('npos', 3)
        self.alimits = kwa.setdefault('alimits', None)
        amin, amax = self._intensity_limits(img, kwa)
        h_in = kwa.pop('h_in', 8)

        kwfig = {}
        _fig=gr.plt.figure(num   = kwa.get('num',None),\
                       figsize   = kwa.get('figsize',(h_in*float(img.shape[1])/img.shape[0], h_in)),\
                       dpi       = kwa.get('dpi',80),\
                       facecolor = kwa.get('facecolor','w'),\
                       edgecolor = kwa.get('edgecolor','w'),\
                       frameon   = kwa.get('frameon',True),\
                       clear     = kwa.get('clear',False),\
                       **kwfig)
                       #FigureClass = kwa.get('FigureClass',Figure)

        kwfica={}
        self.fig, self.axim, self.axcb = gr.fig_img_cbar_axes(\
            fig=_fig,\
            win_axim = kwa.get('win_axim', (0.05,0.03,0.87,0.93)),\
            win_axcb = kwa.get('win_axcb', (0.923,0.03,0.02,0.93)), **kwfica)

        # set default pars for gr.imshow_cbar
        kwic={'amin':amin,
              'amax':amax,
              'extent'       :kwa.get('extent',None),
              'interpolation':kwa.get('interpolation','nearest'),
              'aspect'       :kwa.get('aspect','auto'),
              'origin'       :kwa.get('origin','upper'),
              'orientation'  :kwa.get('orientation','vertical'),
              'cmap'         :kwa.get('cmap','inferno')
              }
        self.imsh, self.cbar = gr.imshow_cbar(self.fig, self.axim, self.axcb, img, **kwic)

        gr.draw_fig(self.fig)

        #gr.show(mode=1)


    def _intensity_limits(self, img, kwa):
        """ returns tuple of intensity limits (amin, amax)
            NOTE: kwa is dict (NOT **kwa) because need to clean dict of parameters
        """
        alimits = kwa.pop('alimits', self.alimits)
        return arr_median_limits(
                   arr  = kwa.pop('arr', img),\
                   nneg = kwa.pop('nneg', self.nneg), 
                   npos = kwa.pop('npos', self.npos))\
               if alimits is None else alimits


    def update(self, img, **kwa):
        """use kwa: arr=arr, nneg=1, npos=3 OR alimits=(amin,amax)
        """
        amin, amax = self._intensity_limits(img, kwa)
        self.imsh.set_data(img)
        self.imsh.set_clim(amin, amax)

        #gr.show(mode=1)

#----
