
import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from psana import DataSource
from psana.hexanode.WFPeaks import WFPeaks
from psana.hexanode.HitFinder import HitFinder

from psana.pyalgos.generic.NDArrUtils import print_ndarr
from psana.pyalgos.generic.Utils import str_kwargs, do_print

#----------

USAGE = 'Use command: python %s [test-number]' % sys.argv[0]

#----------

def proc_data(**kwargs):


    DSNAME       = kwargs.get('dsname', '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2')
    DETNAME      = kwargs.get('detname','tmo_quadanode')
    EVSKIP       = kwargs.get('evskip', 0)
    EVENTS       = kwargs.get('events', 10) + EVSKIP
    VERBOSE      = kwargs.get('verbose', False)

    ds    = DataSource(files=DSNAME)
    orun  = next(ds.runs())

    print('\nruninfo expt: %s  runnum: %d' % (orun.expt, orun.runnum))

    det   = orun.Detector(DETNAME)
    kwargs['consts'] = det.calibconst

    peaks = WFPeaks(**kwargs)
    HF = HitFinder(kwargs)
    
    xs = np.empty([0,])
    ys = np.empty([0,])
    ts = np.empty([0,])        
    sm = {'u':np.empty([0,]),'v':np.empty([0,])}
    sb = {'u':np.empty([0,]),'v':np.empty([0,])}  
    
    for nev,evt in enumerate(orun.events()):

        if nev<EVSKIP : continue
        if nev>EVENTS : break

        t0_sec = time()

        wts = det.raw.times(evt)     
        wfs = det.raw.waveforms(evt)

        nhits, pkinds, pkvals, pktsec = peaks(wfs,wts) # ACCESS TO PEAK INFO
        pktsec = pktsec*1e9
        if VERBOSE :
            print("  waveforms processing time = %.6f sec" % (time()-t0_sec))
            print_ndarr(wfs,    '  waveforms      : ', last=4)
            print_ndarr(wts,    '  times          : ', last=4)
            print_ndarr(nhits,  '  number_of_hits : ')
            print_ndarr(pktsec, '  peak_times_sec : ', last=4)
            
        HF.FindHits(pktsec[4,:nhits[4]],pktsec[0,:nhits[0]],pktsec[1,:nhits[1]],pktsec[2,:nhits[2]],pktsec[3,:nhits[3]])
        xs1,ys1,ts1 = HF.GetXYT()
        xs = np.concatenate([xs,xs1],axis=0)
        ys = np.concatenate([ys,ys1],axis=0)
        ts = np.concatenate([ts,ts1],axis=0)     
        
        print(str(nev)+'th event in 950 total events')    
        
    xbins = np.linspace(-50,50,200)
    tbins = np.linspace(1500,4000,1250)    
    plt.figure()
    
    plt.subplot(121)
    xy,_,_ = np.histogram2d(xs,ys,bins=[xbins,xbins])  
    im = plt.imshow(xy+1,extent=[xbins[0],xbins[-1],xbins[0],xbins[-1]],norm=LogNorm(vmin=1, vmax=xy.max()+1))
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('X - Y')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)') 
    
    plt.subplot(122)
    ht,_ = np.histogram(ts,bins=tbins)  
    plt.plot(tbins[:-1],ht)
    plt.title('Time of Flight')
    plt.xlabel('T (ns)')
    plt.ylabel('Yield (arb. units)')    
    
    plt.tight_layout()
    plt.show()
    
#----------

if __name__ == "__main__" :


    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print('%s\nTEST %s' % (50*'_', tname))

    kwargs = {'dsname'   : '/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e001000.xtc2',
              'detname'  : 'tmo_quadanode',
              'numchs'   : 5,
              'numhits'  : 16,
              'evskip'   : 0,
              'events'   : 950,
              'ofprefix' : './',
              'run'      : 100,
              'exp'      : 'amox27716',
              'verbose'  : False,
              
              'runtime_u' : 90,
              'runtime_v' : 100,
              'tsum_avg_u' : 130,
              'tsum_avg_v' : 141,
              'tsum_hw_u' : 6,
              'tsum_hw_v' : 6,
              'f_u' : 1,
              'f_v' : 1,
              'Rmax': 45,

              'version'  : 4,
              'DLD': True,              
              'paramsCFD' : {0: {'channel': 'mcp',
                              'delay': 3.068e-09,
                              'fraction': 0.35,
                              'offset': 0.054470544805354439,
                              'polarity': 'Negative',
                              'sample_interval': 2.5e-10,
                              'threshold': 0.056374120466532174,
                              'timerange_high': 1e-05,
                              'timerange_low': 1e-06,
                              'walk': 0},
                              1: {'channel': 'x1',
                              'delay': 3.997500000000001e-09,
                              'fraction': 0.35,
                              'offset': 0.032654320557811034,
                              'polarity': 'Negative',
                              'sample_interval': 2.5e-10,
                              'threshold': 0.048439800379417808,
                              'timerange_high': 1e-05,
                              'timerange_low': 1e-06,
                              'walk': 0},
                             2: {'channel': 'x2',
                               'delay': 4.712500000000001e-09,
                              'fraction': 0.35,
                              'offset': 0.058295909692775157,
                              'polarity': 'Negative',
                              'sample_interval': 2.5e-10,
                              'threshold': 0.062173077232695384,
                              'timerange_high': 1e-05,
                              'timerange_low': 1e-06,
                              'walk': 0},
                             3: {'channel': 'y1',
                              'delay': 4.5435e-09,
                              'fraction': 0.35,
                              'offset': 0.01740340726630819,
                              'polarity': 'Negative',
                              'sample_interval': 2.5e-10,
                              'threshold': 0.035850750860370109,
                              'timerange_high': 1e-05,
                              'timerange_low': 1e-06,
                              'walk': 0},
                             4: {'channel': 'y2',
                              'delay': 4.140500000000001e-09,
                              'fraction': 0.35,
                              'offset': 0.0088379291811293368,
                              'polarity': 'Negative',
                              'sample_interval': 2.5e-10,
                              'threshold': 0.035254198205580331,
                              'timerange_high': 1e-05,
                              'timerange_low': 1e-06,
                              'walk': 0}}}
                              
    cfdpars= {'cfd_base'       :  0.,
              'cfd_thr'        : -0.05,
              'cfd_cfr'        :  0.85,
              'cfd_deadtime'   :  10.0,
              'cfd_leadingedge':  True,
              'cfd_ioffsetbeg' :  1000,
              'cfd_ioffsetend' :  2000,
              'cfd_wfbinbeg'   :  6000,
              'cfd_wfbinend'   : 22000,
             }

    kwargs.update(cfdpars)                              

    proc_data(**kwargs)

    print('\n%s' % USAGE)
    sys.exit('End of %s' % sys.argv[0])
