import pyqtgraph as pg
import numpy
import argparse
import psdaq.seq.seq

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

    def frame_number(self):
        return int(self.acframe) if self.acmode else int(self.frame)

class SeqUser:
    def __init__(self, start=0, stop=200, acmode=False):
        global f
        self.start   = start
        self.stop    = stop
        self.acmode  = acmode
        print('start, stop: {:},{:}'.format(start,stop))
        
        self.app  = pg.Qt.QtGui.QApplication([])
        self.win  = pg.GraphicsWindow()
        self.q    = self.win.addPlot(title='Trigger Bits',data=[0],row=0, col=0)
        self.q.getAxis('left').setLabel('bits')
        self.q.getAxis('bottom').setLabel('AC timeslots' if acmode else 'MHz timeslots')
        self.q.showAxis('right')
        self.q.getAxis('top').setLabel('time','sec')
        self.q.showAxis('top')
        self.q.getAxis('top').setScale((1/360) if acmode else (1400/1300e6))
        self.plot  = self.q.plot(pen=None, symbol='s', pxMode=True, size=2)
        self.xdata = []
        self.ydata = []

    def execute(self, title, instrset, descset):

        x = 0

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
                            self.xdata.append(frame)
                            self.ydata.append(i)
                frame   = engine.frame_number()
                request = int(engine.request)

        self.plot.setData(self.xdata,self.ydata)

        self.q.setTitle(title)
        self.q.showGrid(x=True,y=False,alpha=1.0)

        ticks = []
        for i,label in enumerate(descset):
            ticks.append( (float(i), label) )
        ax = self.q.getAxis('right')
        ax.setTicks([ticks])

        self.app.processEvents()

        if engine.modes == 3:
            print(bcolors.WARNING + "Found both fixed-rate-sync and ac-rate-sync instructions." + bcolors.ENDC)

        input(bcolors.OKGREEN+'Press ENTER to exit'+bcolors.ENDC)

def main():
    parser = argparse.ArgumentParser(description='simple sequence plotting gui')
    parser.add_argument("seq", help="sequence script to plot")
    parser.add_argument("--start", default=  0, type=int, help="beginning timeslot")
    parser.add_argument("--stop" , default=200, type=int, help="ending timeslot")
    parser.add_argument("--mode" , default='CW', help="timeslot mode [CW,AC]")
    args = parser.parse_args()
    
    config = {'title':'TITLE', 'descset':None, 'instrset':None}

    exec(compile(open(args.seq).read(), args.seq, 'exec'), {}, config)

    seq = SeqUser(start=args.start,stop=args.stop,acmode=(args.mode=='AC'))
    seq.execute(config['title'],config['instrset'],config['descset'])

if __name__ == 'main':
    main()
