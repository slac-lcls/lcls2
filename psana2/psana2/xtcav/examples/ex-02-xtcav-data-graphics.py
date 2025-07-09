#!/usr/bin/env python
"""
"""
from psana2.xtcav.examples.ex_utils import data_file, sys
from psana2.pyalgos.generic.NDArrUtils import print_ndarr
from psana import DataSource
import psana2.pyalgos.generic.Graphics as gr

print('e.g.: [python] %s [test-number]' % sys.argv[0])

#----------

def test_xtcav_data_access() :
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'

    #fig, axim = fig_axis()
    fig, axim, axcb = gr.fig_img_cbar_axes(fig=None,\
             win_axim=(0.05,  0.05, 0.87, 0.93),\
             win_axcb=(0.923, 0.05, 0.02, 0.93)) #, **kwargs)

    ds = DataSource(files=data_file(tname))
    orun = next(ds.runs())
    det = orun.Detector('xtcav')

    print('test_xtcav_data    expt: %s runnum: %d\n' % (orun.expt, orun.runnum))

    for nev,evt in enumerate(orun.events()):
        if nev>10 : break
        print('Event %03d'%nev, end='')

        nda = det.raw(evt)
        print_ndarr(nda, '  det.raw(evt):')

        mean, std = nda.mean(), nda.std()
        aran = (mean-3*std, mean+5*std)

        axim.clear()
        axcb.clear()

        imsh = gr.imshow(axim, nda, amp_range=aran, extent=None, interpolation='nearest',\
                         aspect='auto', origin='upper', orientation='horizontal', cmap='inferno')
        cbar = gr.colorbar(fig, imsh, axcb, orientation='vertical', amp_range=aran)

        gr.set_win_title(fig, 'Event: %d' % nev)
        gr.draw_fig(fig)
        gr.show(mode='non-hold')

    gr.save_fig(fig, prefix='./img-%s-r%04d-e%06d-'%(orun.expt, orun.runnum, nev), suffix='.png', )
    gr.show()

#----------

if __name__ == "__main__":
    test_xtcav_data_access()

#----------
