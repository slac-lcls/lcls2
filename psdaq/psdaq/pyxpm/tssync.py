from datetime import datetime
from psdaq.pyxpm.pvhandler import *

class TsSync():
    def __init__(self,name,reg):
        self.reg = reg
        self.ts  = None

        self.pv_skew   = addPV(name+':TsSkew'  ,'f',0.)

        #  Set the default clock rate
        self.reg.CountInterval.set(0x05050d)

        #  Fetch the time
        tsec = (datetime.now() - datetime(1990,1,1)).total_seconds()
        ts = (int(tsec)<<32) | int(1.e9*(tsec%1.0))
        pid = int(tsec * 910000*14./13)
        self.reg.TStampWr  .set(ts)
        self.reg.TStampSet .set(0)
        self.reg.PulseIdWr .set(pid)
        self.reg.PulseIdSet.set(0)

    def update(self):
        ts  = self.reg.TStampRd.get()

        if self.ts is None:
            self.ts  = ts
        else:
            #  1 second has elapsed
            #  How far are we from the expected time?
            #  If we are out of the window, change the freq
            dsec = (ts - self.ts)/float(1<<32) - 1.0
            if dsec < -0.02:
                self.reg.CountInterval.set(0x050c1f)
            elif dsec > 0.02:
                self.reg.CountInterval.set(0x050815)
            else:
                self.reg.CountInterval.set(0x05050d)
            self.ts = ts
            self.dsec = dsec

            pvUpdate(self.pv_skew,dsec)


