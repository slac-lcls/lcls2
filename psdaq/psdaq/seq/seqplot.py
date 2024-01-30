from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
import argparse
import psdaq.seq.seq
from psdaq.seq.globals import *

f=None
verbose=False

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Engine(object):

    def __init__(self, acmode=False):
        self.request = 0
        self.instr   = 0
        self.frame   = -1  # 1MHz timeslot
        self.acframe = -1  # 360Hz timeslot
        self.acmode  = acmode
        self.modes   = 0
        self.ccnt    = [0]*4
        self.done    = False
        self.returnaddr = None

    def frame_number(self):
        return int(self.acframe) if self.acmode else int(self.frame)

    def __str__(self):
        return f'request {self.request}  instr {self.instr}  returnaddr {self.returnaddr}  frame {self.frame}  ccnt {self.ccnt}'

class SeqUser(object):
    def __init__(self, start=0, stop=200, acmode=False):
        global f
        self.start   = start
        self.stop    = stop
        self.acmode  = acmode
        print('start, stop: {:},{:}'.format(start,stop))
        
        self.xdata = []
        self.ydata = []

    def execute(self, title, instrset, descset):

        x = 0

        # keep separate lists for each request line and merge at the end
        xdata = {}
        engine  = Engine(self.acmode)
        while engine.frame_number() < self.stop and not engine.done:

            frame   = engine.frame_number()
            request = int(engine.request)

            instrset[engine.instr].execute(engine)
            if engine.frame_number() != frame:
                if verbose:
                    print('frame: {}  instr {}  request {:x}'.format
                          (frame,engine.instr,request))
                if frame < self.start:
                    continue
                if request != 0:
                    for i in range(16):
                        if (request&(1<<i)):
                            if i in xdata:
                                xdata[i].append(frame)
                            else:
                                xdata[i]=[frame]
                frame   = engine.frame_number()
                request = int(engine.request)

        for i in range(16):
            if i in xdata:
                self.xdata.extend(xdata[i])
                self.ydata.extend([i]*len(xdata[i]))

        print(f'engine exited {engine}')

        if engine.modes == 3:
            print(bcolors.WARNING + "Found both fixed-rate-sync and ac-rate-sync instructions." + bcolors.ENDC)

class PatternWaveform(object):
    def __init__(self):
        self.gl = pg.GraphicsLayoutWidget()
        self.index = 0
        self.q0 = None

    def _color(idx,nidx):
        x = float(idx)/float(nidx-1)
        c = (0,0,0)
        if x < 0.5:
            c = (511*(0.5-x),511*x,0)
        else:
            c = (0,511*(1.0-x),511*(x-0.5))
        return c

    def add(self, title, xdata, ydata):
        #  Plotting lots of consecutive buckets with scatter points is
        #  time consuming.  Replace consecutive points with a line.
        def plot(q, x, y):
            if len(x):
                rx = []
                ry = []
                bfirst = x[0]
                bnext  = bfirst+1
                dlast  = y[0]
                for i in range(1,len(x)):
                    b = x[i]
                    if b==bnext and y[i]==dlast:
                        bnext = b+1
                    elif bnext-bfirst > 1:
                        q.plot([bfirst,bnext-1],[dlast,dlast],pen=pg.mkPen('w',width=5))
                        dlast  = y[i]
                        bfirst = b
                        bnext  = b+1
                    else:
                        rx.append(bfirst)
                        ry.append(dlast)
                        dlast  = y[i]
                        bfirst = b
                        bnext  = b+1
                if bnext-bfirst > 1:
                    q.plot([bfirst,bnext-1],[dlast,dlast],pen=pg.mkPen('w',width=5))
                else:
                    rx.append(bfirst)
                    ry.append(dlast)
                q.plot(rx, ry, pen=None,
                       symbolBrush=(255,255,255),
                       symbol='s',pxMode=True, size=2)

        buckets = xdata
        dests   = ydata

        q0 = self.gl.addPlot(col=0,row=self.index)
        q0.setLabel('left'  ,title )
        q0.showGrid(True,True)
        ymax = np.amax(dests,initial=0)
        ymin = np.amin(dests,initial=288)

        plot(q0,buckets,dests)
        q0.setRange(yRange=[ymin-0.5,ymax+0.5])

        if self.q0 is None:
            self.q0 = q0
        else:
            q0.setXLink(self.q0)

        self.index += 1

def main():
    parser = argparse.ArgumentParser(description='simple sequence plotting gui')
    parser.add_argument("--seq", required=True, nargs='+', type=str, help="sequence engine:script pairs; e.g. 0:train.py")
    parser.add_argument("--time", required=False, type=float, default=1., help="simulated time (sec)")
    args = parser.parse_args()

    config = {'title':'TITLE', 'descset':None, 'instrset':None}

    app = QtWidgets.QApplication([])
    plot = PatternWaveform()

    for f in args.seq:
        engine,fname = f.split(':')[:2]
        print(f'eng {engine} fn {fname}')

        seq = 'from psdaq.seq.seq import *\n'
        seq += open(fname).read()
        exec(compile(seq, fname, 'exec'), {}, config)

        for i,ins in enumerate(config['instrset']):
            print(f'{i}: {ins}')

        seq = SeqUser(start=0,stop=int(args.time*TPGSEC),acmode=False)
        seq.execute(config['title'],config['instrset'],config['descset'])

        ydata = np.array(seq.ydata)+int(engine)*4+272
        plot.add(fname, seq.xdata, ydata)

    MainWindow = QtWidgets.QMainWindow()
    centralWidget = QtWidgets.QWidget(MainWindow)
    vb = QtWidgets.QVBoxLayout()
    vb.addWidget(plot.gl)
    centralWidget.setLayout(vb)
    MainWindow.setCentralWidget(centralWidget)
    MainWindow.updateGeometry()
    MainWindow.show()

    app.exec_()

if __name__ == '__main__':
    main()
