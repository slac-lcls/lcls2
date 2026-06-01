"""  
     Setup is trigger at 70 kHz
     MONTRIG counters are 20 bits
     Scan phase and collect: ontime, early, late
     Pause triggers at each step (DAQ:NEH:XPM:2:PART:6:Run)
     Do not reconfigure at each step.  We want to see the phase move from
     one clock to the next.  Each step is 1/56 * 70/13 (ns) * 1/6 = 16 ps.
"""

import sys
import argparse
import time

import pyqtgraph as pg
from PyQt5.QtGui import *
from multiprocessing import Process, Queue

from psdaq.cas.pvedit import *
from p4p.client.thread import Context
from psdaq.hsd.pvdef import *

que       = None
psSet     = None
l0Enable  = None
l0Disable = None
monTrig   = None

def analyze(q):

    app = pg.Qt.QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget()
    win.setBackground('w')

    pp = win.addPlot(data=[0],row=0,col=0)
    pp.setTitle('VCO Phase (ns)')
    earlyplot = pp.plot(pen=pg.mkPen(color=(255,0,0)))
    lateplot  = pp.plot(pen=pg.mkPen(color=(0,0,255)))

    ph = []
    de = []
    dl = []

    ns_per_step = 1/56 * 70/13 / 6

    while True:
        total, early, late, phase = q.get()

        print(f'recv total {total}')
        ph.append(phase*ns_per_step)
        de.append(early/total)
        dl.append(late /total)

        earlyplot.setData(ph,de)
        lateplot .setData(ph,dl)

        win.show()

        app.processEvents()


def callback(err):

    global que

    value = monTrig.__value__
    print(f'callback {value}')

    total  = value['ontime']+value['early']+value['late']
    print(f'total {total}')
    if total > args.period:
        que.put((total,value['early'],value['late'],value['phase']))

        l0Enable .put(0)
        l0Disable.put(1<<args.group)

        psVal = psSet.get()
        psVal['mmcmsetup'] = 1
        psVal['mmcmphase'] = value['phase']+args.range[2]
        psSet.put(psVal)

        if value['phase'] < args.range[1]:
            l0Enable .put(1<<args.group)
            l0Disable.put(0)
        
def main():
    global args
    global que
    global psSet
    global l0Enable
    global l0Disable
    global monTrig

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("--base", help="PV base", default='DAQ:TMO:HSD:2_41:A')
    parser.add_argument("--xpm",  help="XPM PV", default='DAQ:NEH:XPM:2')
    parser.add_argument("--group",help="readout group", default=6)
    parser.add_argument("--period", help="period", type=float, default=140000)
    parser.add_argument("--range", help="range", type=int, nargs=3, default=(0,56,8))
    parser.add_argument("--verbose", help="verbose msg", action='store_true', default=False)
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    que = Queue()
    proc = Process(target=analyze, args=(que,))
    proc.start()

    psSet     = Pv(f'{args.base}:RESET'  ,isStruct=True)
    l0Enable  = Pv(f'{args.xpm}:GroupL0Enable')
    l0Disable = Pv(f'{args.xpm}:GroupL0Disable')

    psVal = psSet.get()
    psVal['mmcmsetup'] = 1
    psVal['mmcmphase'] = args.range[0]
    psSet.put(psVal)

    monTrig = Pv(f'{args.base}:MONTRIG',callback=callback,isStruct=True)
    
    l0Enable .put(1<<args.group)
    l0Disable.put(0)

    while(True):
        time.sleep(1)

    proc.join()

if __name__ == '__main__':
    main()
