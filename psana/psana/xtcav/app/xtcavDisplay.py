#!/usr/bin/env python

import sys
import argparse
import logging
logger = logging.getLogger(__name__)
from psana.pyalgos.generic.Utils import init_logger, STR_LEVEL_NAMES, input_single_char

scrname = sys.argv[0].rsplit('/')[-1]

usage = '\nE.g. : %s amox23616 137' % scrname\
      + '\n  or : %s amox23616 137 -l DEBUG -f fname.xtc2\n' % scrname
print(usage)

d_fname = '/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0137-e000100-xtcav-v2.xtc2'

parser = argparse.ArgumentParser(description='XTCAV DISPLAY results of data processing') # , usage=usage())
parser.add_argument('experiment', help='psana experiment string (e.g. "amox23616")')
parser.add_argument('run', type=int, help="run number")
parser.add_argument('-f', '--fname', type=str, default=d_fname, help='xtc2 file')
parser.add_argument('-l', '--loglev', default='DEBUG', type=str, help='logging level name, one of %s' % STR_LEVEL_NAMES)
parser.add_argument('-p', '--pause', type=float, default=2, help="pause [sec] to browse events")
parser.add_argument('-g', '--grmode', type=int, default=1, help="graphics control mode; 0-close window, 1-using keyboard keys")
parser.add_argument('-n', '--nevents', type=int, default=100, help="number of non-empty events")

args = parser.parse_args()
print('Arguments of type %s as %s' % (type(args), type(vars(args))))
for k,v in vars(args).items() : print('  %12s : %s' % (k, str(v)))

init_logger(args.loglev, fmt='[%(levelname).1s] L%(lineno)04d : %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')

#----------
#----------
#----------
#----------

import matplotlib.pyplot as plt
###plt.switch_backend('Qt5Agg')

import numpy as np

from psana import DataSource
from psana.xtcav.LasingOnCharacterization import LasingOnCharacterization, cons, setDetectors
from psana.pyalgos.generic.NDArrUtils import info_ndarr, print_ndarr
import psana.xtcav.UtilsPsana as xtup


def getLasingOffShot(lon, fname_loff):
    results=lon._pulse_characterization
    loff = lon._lasingoffreference
    ibunch = 0

    group = results.groupnum[ibunch]
    profs = loff.averaged_profiles

    ds = DataSource(files=fname_loff)
    run = next(ds.runs())

    print('XXXX dir(run)', dir(run))
    print('XXXX max_events', run.max_events)
    print('XXXX esm', run.esm)
    print('XXXX timestamp', run.timestamp)
    print('XXXX dm', run.dm)
    print('XXXX smd_dm', run.smd_dm)
    #====================
    sys.exit('TEST EXIT')
    #====================

    camera = run.Detector(cons.DETNAME)
    camraw = xtup.get_attribute(camera,'raw')

    #LCLS1 stuff does not work here
    #times = run.times()
    #time = profs.eventTime[ibunch][group]
    #fid = profs.eventFid[ibunch][group]
    #et = EventTime(int(time),int(fid))
    #evt_loff = run.event(et)

    if camraw is None:
        print('No lasing off image found for unixtime',time,'and fiducials',fid)
    print('Found lasing off shot in run',loff.parameters.run)
    return camraw(evt_loff)


def figaxtitles(fig=None):
    """
       fig, axes, titles = figaxtitles()
       ax11, ax12, ax21, ax22, ax31, ax32, axcb11, axcb12 = axes
    """
    w,h, wcb = 0.40, 0.26, 0.01
    x1,x2 = 0.05, 0.55
    y1,y2,y3 = 0.69, 0.36, 0.03

    _fig = fig if fig is not None else\
           plt.figure(figsize=(8,7), dpi=100, facecolor='w', edgecolor='w')#, frameon=True)
    axes = (\
     _fig.add_axes((x1,y1,w,h)),
     _fig.add_axes((x2,y1,w,h)),
     _fig.add_axes((x1,y2,w,h)),
     _fig.add_axes((x2,y2,w,h)),
     _fig.add_axes((x1,y3,w,h)),
     _fig.add_axes((x2,y3,w,h)),
     _fig.add_axes((0.45,y1,wcb,h)),
     _fig.add_axes((0.95,y1,wcb,h)),
    )
    titles = 'Lasing On', 'Lasing Off', 'Current', 'E (Delta)', 'E (Sigma)', 'Power'
 
    return _fig, axes, titles

class Control :
    PAUSE = False

CONTROL = Control

def press(event):
    print('pressed %s of possible e-exit, h/p/d-hold/pause/delay, c/g-continue/go' % event.key)
    sys.stdout.flush()
    #plt.ion()
    #plt.show()
    ch = event.key.lower()
    if   ch == 'e': sys.exit('Terminated from keyboard')
    elif ch in ('h','p','d',) :
        print('Pause is set')
        CONTROL.PAUSE = True
    elif ch in ('c','g') : 
        print('Continue event loop')
        CONTROL.PAUSE = False
    elif event.key == 'x': return


def procEvents(args):

    fname      = getattr(args, 'fname',      '/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0137-e000100-xtcav-v2.xtc2')
    fname_loff = getattr(args, 'fname_loff', '/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0131-e000200-xtcav-v2.xtc2')
    nevents    = getattr(args, 'nevents', 100)
    mode       = getattr(args, 'mode', 'smd')
    exp        = getattr(args, 'experiment', None)
    grmode     = getattr(args, 'grmode', 1)
    pause      = getattr(args, 'pause', 1)

    ds = DataSource(files=fname)
    run = next(ds.runs())

    lon = LasingOnCharacterization(args, run, setDetectors(run))

    camraw   = lon._camraw
    valsebm  = lon._valsebm
    valsgd   = lon._valsgd
    valseid  = lon._valseid
    valsxtp  = lon._valsxtp

    if grmode == 1 :
        fig, axes, titles = figaxtitles()
        ax11, ax12, ax21, ax22, ax31, ax32, axcb11, axcb12 = axes
        plt.ion() # do not hold control on plt.show()
        fig.canvas.mpl_connect('key_press_event', press)

    nimgs=0
    for nev,evt in enumerate(run.events()):

        raw = camraw(evt)
        logger.info('Event %03d' % nev)
        logger.debug(info_ndarr(raw, 'camera raw:'))
        if raw is None: continue

        if not lon.processEvent(evt): continue

        nimgs += 1
        if nimgs>=nevents: break

        time, power, agreement, pulse = lon.resultsProcessImage()
        #time, power = lon.xRayPower(method="COM") 
        #agreement = lon.reconstructionAgreement()

        print('%sAgreement:%7.3f%%  Max power: %g  GW Pulse Delay: %.3f '%(12*' ', agreement*100,np.amax(power), pulse[0]))

        #gd = valsgd(evt)
        f_11_ENRC = 'N/A' if valsgd is None else valsgd.f_11_ENRC(evt)
        print('Agreement:', agreement, 'Gasdet.f_11_ENRC:', f_11_ENRC)
        
        if agreement<0.5: continue
    
        results=lon._pulse_characterization

        #raw_off = getLasingOffShot(lon, fname_loff)
        profiles = lon._lasingoffreference.averaged_profiles
        #print('XXX profiles.eCOMslice', profiles.eCOMslice)

        if grmode == 0 :
            fig, axes, titles = figaxtitles()
            ax11, ax12, ax21, ax22, ax31, ax32, axcb11, axcb12 = axes

        for ax in axes : ax.cla()
        for ax, title in zip(axes[:6], titles) : ax.set_title(title, color='k', fontsize=12)

        #img_loff = profiles.eCOMslice[0]; ax12.set_title('Lasing Off: eCOMslice', color='k', fontsize=12)
        #img_loff = profiles.eRMSslice[0]; ax12.set_title('Lasing Off: eRMSslice', color='k', fontsize=12)
        img_loff = profiles.eCurrent[0]; ax12.set_title('Lasing Off: eCurrent', color='k', fontsize=12)

        imsh11 = ax11.imshow(raw, interpolation='nearest', aspect='auto', origin='upper', extent=None, cmap='inferno')
        cbar11 = fig.colorbar(imsh11, cax=axcb11, orientation='vertical')

        imsh12 = ax12.imshow(img_loff, interpolation='nearest', aspect='auto', origin='upper', extent=None, cmap='inferno')
        cbar12 = fig.colorbar(imsh12, cax=axcb12, orientation='vertical')

        ax21.plot(time[0],results.lasingECurrent[0],label='lasing')
        ax21.plot(time[0],results.nolasingECurrent[0],label='nolasing')

        ax22.plot(time[0],results.lasingECOM[0],label='lasing')
        ax22.plot(time[0],results.nolasingECOM[0],label='nolasing')

        ax31.plot(time[0],results.lasingERMS[0],label='lasing')
        ax31.plot(time[0],results.nolasingERMS[0],label='nolasing')

        ax32.plot(time[0],power[0])

        # LCLS1
        #plt.subplot(3,2,1)
        #plt.title('Lasing On')
        #plt.imshow(raw)

        #xtcav_lasingoff = getLasingOffShot(lon, fname_loff)
        #plt.subplot(3,2,2)
        #plt.title('Lasing Off')
        #plt.imshow(xtcav_lasingoff)
    
        #plt.subplot(3,2,3)
        #plt.title('Current')
        #plt.plot(time[0],results.lasingECurrent[0],label='lasing')
        #plt.plot(time[0],results.nolasingECurrent[0],label='nolasing')
        ##plt.legend()
    
        #plt.subplot(3,2,4)
        #plt.title('E (Delta)')
        #plt.plot(time[0],results.lasingECOM[0],label='lasing')
        #plt.plot(time[0],results.nolasingECOM[0],label='nolasing')
        ##plt.legend()
    
        #plt.subplot(3,2,5)
        #plt.title('E (Sigma)')
        #plt.plot(time[0],results.lasingERMS[0],label='lasing')
        #plt.plot(time[0],results.nolasingERMS[0],label='nolasing')
        ##plt.legend()
    
        #plt.subplot(3,2,6)
        #plt.title('Power')
        #plt.plot(time[0],power[0])

        fig.canvas.set_window_title('Event %3d good %3d' % (nev, nimgs))

        #fig.canvas.draw()
        #plt.show(block=False) 
        plt.draw()

        if grmode == 0 :
            print('\nConntinue - close graphics window (click on [x] on window frame)')
            plt.show()

        elif grmode == 1 :
            #print('\nPAUSE %s: ' % CONTROL.PAUSE)
            print('\nControl keys: e-exit, h/p/d-hold/pause/delay, c/g-continue/go')
            plt.pause(pause) # hack to make it work... othervise show() does not work...
            while CONTROL.PAUSE : plt.pause(1)

        #ch = input_single_char('Next event? [y/n]')
        #if ch == 'y': pass
        #else        : sys.exit('Exit by request')
            
    plt.ioff() # hold contraol at show(); plt.ion() is set on any keyboard key press event
    plt.show()



# available quantities from step3, from xtcav/Utils.py:ProcessLasingSingleShot

# 't':t,                                  #Master time vector in fs
# 'powerECOM':powerECOM,                  #Retrieved power in GW based on ECOM
# 'powerERMS':powerERMS,                  #Retrieved power in GW based on ERMS
# 'powerAgreement':powerAgreement,        #Agreement between the two intensities
# 'bunchdelay':bunchdelay,                #Delay from each bunch with respect to the first one in fs
# 'bunchdelaychange':bunchdelaychange,    #Difference between the delay from each bunch with respect to the first one in fs and the same form the non lasing reference
# 'xrayenergy':shotToShot['xrayenergy'],  #Total x-ray energy from the gas detector in J
# 'lasingenergyperbunchECOM': eBunchCOM,  #Energy of the XRays generated from each bunch for the center of mass approach in J
# 'lasingenergyperbunchERMS': eBunchRMS,  #Energy of the XRays generated from each bunch for the dispersion approach in J
# 'bunchenergydiff':bunchenergydiff,             #Distance in energy for each bunch with respect to the first one in MeV
# 'bunchenergydiffchange':bunchenergydiffchange, #Comparison of that distance with respect to the no lasing
# 'lasingECurrent':lasingECurrent,        #Electron current for the lasing trace (In #electrons/s)
# 'nolasingECurrent':nolasingECurrent,    #Electron current for the no lasing trace (In #electrons/s)
# 'lasingECOM':lasingECOM,                #Lasing energy center of masses for each time in MeV
# 'nolasingECOM':nolasingECOM,            #No lasing energy center of masses for each time in MeV
# 'lasingERMS':lasingERMS,                #Lasing energy dispersion for each time in MeV
# 'nolasingERMS':nolasingERMS,            #No lasing energy dispersion for each time in MeV
# 'NB': NB,                               #Number of bunches
# 'groupnum': groupnum                    #group number of lasing-off shot


#----------

procEvents(args) 
sys.exit('END OF TEST %s' % scrname)

#----------
