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

import pickle

que       = None

class MonTrigPv(object):

    def __init__(self, name, mincount):
        self.name     = name
        self.pv       = Pv(name+':MONTRIG', isStruct=True)
        self.mincount = mincount
        self.done     = False
        self.value    = None
        self.values   = []
        
        def monitor_cb(newval):
            v = self.pv.to_value(newval)
            total = v['ontime']+v['early']+v['late']
            self.done = total > self.mincount
            self.value = v
            
        try:
            self.subscription = pvactx.monitor(self.pv.pvname, monitor_cb)
            self.value = None
        except TimeoutError as e:
            logger.error("Timeout exception connecting to PV %s", self.pv.pvname)

    def latch(self):
#        self.values.append(self.value)
        t = (self.name, self.value['ontime'], self.value['early'], self.value['late'], self.value['phase'])
        print(f'Latched {t}')
        que.put(t)
        
def analyze(q,names,record):

    app = pg.Qt.QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget()
    win.setBackground('w')

    pp = win.addPlot(data=[0],row=0,col=0)
    pp.setTitle('VCO Phase (ns)')

    earlyplot = {}
    lateplot  = {}
    de        = {}
    dl        = {}
    ph        = {}
    
    for n in names:
        earlyplot[n] = pp.plot(pen=pg.mkPen(color=(255,0,0)))
        lateplot [n] = pp.plot(pen=pg.mkPen(color=(0,0,255)))
        de       [n] = []
        dl       [n] = []
        ph       [n] = []

    ns_per_step = 1/56 * 70/13 / 6

    f = None
    if record:
        f = open('hsdPs.pyc','wb')
        
    while True:
        t = q.get()

        if f:
            pickle.dump(t,f)
            
        n      = t[0]
        ontime = t[1]
        early  = t[2]
        late   = t[3]
        phase  = t[4]
        total = ontime + early + late

        print(f'recv total {total}')
        ph[n].append(phase*ns_per_step)
        de[n].append(early/total)
        dl[n].append(late /total)

        earlyplot[n].setData(ph[n],de[n])
        lateplot [n].setData(ph[n],dl[n])

        win.show()

        app.processEvents()


def main():
    global args
    global que

    parser = argparse.ArgumentParser(description='simple pv monitor gui')
    parser.add_argument("--base", help="PV base", default='[DAQ:TMO:HSD:2_41:A]',nargs='+')
    parser.add_argument("--xpm",  help="XPM PV", default='DAQ:NEH:XPM:2')
    parser.add_argument("--group",help="readout group", type=int, default=6)
    parser.add_argument("--period", help="period", type=float, default=140000)
    parser.add_argument("--range", help="range", type=int, nargs=3, default=(0,56,8))
    parser.add_argument("--record", help="record to file", action='store_true', default=False)
    parser.add_argument("--setup", help="set phase and exist", action='store_true', default=False)
    parser.add_argument("--verbose", help="verbose msg", action='store_true', default=False)
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    psSet   = []
    monTrig = []
    for b in args.base:
        psSet  .append(Pv(f'{b}:RESET'  ,isStruct=True))
        monTrig.append(MonTrigPv(b,args.period))

    groups = (1<<args.group) + 1
    l0Enable  = Pv(f'{args.xpm}:GroupL0Enable')
    l0Disable = Pv(f'{args.xpm}:GroupL0Disable')

    psVal = psSet[0].get()
    psVal['mmcmsetup'] = 1

    if args.setup:
        print(f'Setting phase {args.range[0]}')
        psVal['mmcmphase'] = args.range[0]
        for p in psSet:
            p.put(psVal)
        sys.exit(1)
        
    l0Enable .put(0)
    l0Disable.put(groups)
    
    que = Queue()
    proc = Process(target=analyze, args=(que,args.base,args.record))
    proc.start()

    for ph in range(args.range[0],args.range[1],args.range[2]):

        print(f'Setting phase {ph}')
        psVal['mmcmphase'] = ph
        for p in psSet:
            p.put(psVal)

        print(f'Enabling triggers for group 0x{groups:x}')
        l0Enable .put(groups)
        l0Disable.put(0)

        #  Wait until done
        while True:
            print(f'Waiting')
            time.sleep(1)
            done = True
            for m in monTrig:
                done = done and m.done
            if done:
                break;
            
        print(f'Disabling')
        l0Enable .put(0)
        l0Disable.put(groups)

        for m in monTrig:
            m.latch()
        
    #  Dump and analyze
#    app.processEvents()
                       
    proc.join()

if __name__ == '__main__':
    main()
