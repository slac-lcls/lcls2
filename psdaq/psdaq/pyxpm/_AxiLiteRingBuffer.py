#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue AXI-Lite Ring Buffer Module
#-----------------------------------------------------------------------------
# File       : AxiLiteRingBuffer.py
# Created    : 2019-06-24
#-----------------------------------------------------------------------------
# Description:
# PyRogue AXI-Lite Ring Buffer Module
#-----------------------------------------------------------------------------
# This file is part of the rogue software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import datetime
import parse
import pyrogue as pr

class AxiLiteRingBuffer(pr.Device):

    # Last comment added by rherbst for demonstration.
    def __init__(
            self,       
            datawidth        = 32,
            name             = 'AxiLiteRingBuffer',
            description      = 'AXI-Lite Ring Buffer Module',
            **kwargs):
        
        super().__init__(
            name        = name,
            description = description,
            **kwargs)

        self._datawidth = datawidth

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(
            name         = 'blen',
            description  = 'Length of ring buffer',
            offset       = 0x00,
            bitSize      = 20,
            bitOffset    = 0x00,
            base         = pr.UInt,
            mode         = 'RO'
        ))

        self.add(pr.RemoteVariable(   
            name         = 'clear',
            description  = 'Clear buffer',
            offset       = 0x03,
            bitSize      = 1,
            bitOffset    = 0x06,
            base         = pr.UInt,
            mode         = 'RW',
            verify       = False,
        ))

        self.add(pr.RemoteVariable(   
            name         = 'start',
            description  = 'Start buffer',
            offset       = 0x03,
            bitSize      = 1,
            bitOffset    = 0x07,
            base         = pr.UInt,
            mode         = 'RW',
            verify       = False,
        ))

        self.addRemoteVariables(   
            name         = 'data',
            description  = 'Buffer values',
            offset       = 0x4,
            bitSize      = 32,
            bitOffset    = 0x00,
            base         = pr.UInt,
            mode         = 'RO',
            number       = 0x3ff,
            stride       = 4,
            hidden       = True,
        )

        @self.command(name='Dump',description='Dump buffer')
        def _Dump():
            mask  = (1<<self._datawidth)-1
            cmask = (self._datawidth+3)/4
            len_  = self.blen.get()
            if len_ > 0x3ff: 
                len_ = 0x3ff

            buff = []
            for i in range(len_):
                buff.append( self.data[i].get() & mask )

            fmt = '{:0%d'%(self._datawidth/4)+'x} '
            for i in range(len_):
                print(fmt.format(buff[i]),end='')
                if (i&0xf)==0xf:
                    print()
            if (len_&0xf)!=0:
                print()

            i=0
            while i<len_:
                if buff[i]==0x1b5f7:
                    if i+FRAMELEN < len_:
                        ts  = buff[i+2] + (buff[i+3]<<16) + (buff[i+4]<<32) + (buff[i+5]<<48)
                        pid = buff[i+6] + (buff[i+7]<<16) + (buff[i+8]<<32) + (buff[i+9]<<48)
                        print(f'ts [{ts}:016x]  pid [{pid}:016x]  rates [{buff[i+10]}]')
                        gl0a     = ''
                        gl0tag   = ''
                        gl0count = ''
                        for g in range(8):
                            gl0a   += f'{buff[i+x]&1:08x}'
                            gl0tag += f'{(buff[i+x]>>1)&0x1f:08x}'
                            gcount += f'{(buff[i+x+1])+((buff[i+x+2]&0xff)<<16) :08x}'
                        print(f'l0a     : {gl0a}')
                        print(f'l0tag   : {gl0tag}')
                        print(f'l0count : {gl0count}')
                        i += FRAMELEN
                    else:
                        break
                else:
                    i += 1

                        
