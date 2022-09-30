#!/usr/bin/env python3
##############################################################################
## This file is part of 'EPIX'.
## It is subject to the license terms in the LICENSE.txt file found in the 
## top-level directory of this distribution and at: 
##    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
## No part of 'EPIX', including this file, 
## may be copied, modified, propagated, or distributed except according to 
## the terms contained in the LICENSE.txt file.
##############################################################################

import pyrogue as pr
import time

class Si570(pr.Device):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.addRemoteVariables(
            name         = 'Regs',
            description  = 'Registers',
            offset       = (0 << 2),
            bitSize      = 8,
            mode         = 'RW',
            number       = 256,
            stride       = 4,
            verify       = False,
#            hidden       = True,
            hidden       = False
        )

        self.add(pr.LocalCommand(name='program',
                                 description='[119MHz, 185.7MHz]',
                                 function=self._program))

    def reset(self):
        print('reset')
        v = self.Regs[135].get() | 1
        self.Regs[135].set(v)
        v = 1
        while (v&1):
            time.sleep(1.e-3)
            v = self.Regs[135].get()
        print('reset complete')

    def read(self):
        #  Read factory calibration for 156.25 MHz
        hsd_divn = [4,5,6,7,0,9,0,11]
        v = self.Regs[7].get()
        print('Si570[7] = {:x}'.format(v))
        hs_div = hsd_divn[(v>>5)&7]
        n1 = (v&0x1f)<<2
        v = self.Regs[8].get()
        print('Si570[8] = {:x}'.format(v))
        n1 |= (v>>6)&3
        rfreq = v&0x3f
        for i in range(9,13):
            v = self.Regs[i].get()
            print('Si570[{:x}] = {:x}'.format(i,v))
            rfreq <<= 8
            rfreq |= (v&0xff)

        f = (156.25 * float(hs_div * (n1+1))) * float(1<<28)/ float(rfreq)
        print('Read: hs_div {:x}  n1 {:x}  rfreq {:x}  f {:f} MHz'.format(hs_div,n1,rfreq,f))
        return f

    def _program(self):
        self.enable.set(True)
        self._programClk([1])
        self.enable.set(False)

    def _programClk(self,args):

        print(f'program {args}')

        fcal = self.read()

        _hsd_div = [ 7, 3 ]
        _n1      = [ 3, 3 ]
        _rfreq   = [ 5236., 5200. ]
        self.reset();

        fcal = self.read()

        #  Program for 1300/7 MHz

        #  Freeze DCO
        v = self.Regs[137].get() | (1<<4)
        self.Regs[137].set(v)

        index  = args[0]
        hs_div = _hsd_div[index]
        n1     = _n1     [index]
        rfreq  = int(_rfreq[index] / fcal * float(1<<28));

        self.Regs[7].set( ((hs_div&7)<<5) | ((n1>>2)&0x1f) )
        self.Regs[8].set( ((n1&3)    <<6) | ((rfreq>>32)&0x3f) )
        self.Regs[9].set( (rfreq>>24)&0xff )
        self.Regs[10].set( (rfreq>>16)&0xff )
        self.Regs[11].set( (rfreq>>8)&0xff )
        self.Regs[12].set( (rfreq>>0)&0xff )
  
        print('Wrote: hs_div {:x}  n1 {:x}  rfreq {:x}  f {:f} MHz'.format(hs_div, n1, rfreq, fcal))

        #  Unfreeze DCO
        v = self.Regs[137].get() & ~(1<<4)
        self.Regs[137].set(v)

        v = self.Regs[135].get() | (1<<6)
        self.Regs[135].set(v)



