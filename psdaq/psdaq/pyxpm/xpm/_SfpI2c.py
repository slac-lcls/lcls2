#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue _SfpI2c Module
#-----------------------------------------------------------------------------
# File       : _SfpI2c.py
# Created    : 2017-01-17
# Last update: 2017-01-17
#-----------------------------------------------------------------------------
# Description:
# PyRogue _SfpI2c Module
#-----------------------------------------------------------------------------
# This file is part of 'SLAC Firmware Standard Library'.
# It is subject to the license terms in the LICENSE.txt file found in the 
# top-level directory of this distribution and at: 
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
# No part of 'SLAC Firmware Standard Library', including this file, 
# may be copied, modified, propagated, or distributed except according to 
# the terms contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue as pr
import struct

class SfpI2c(pr.Device):
    def __init__(self,       
            name          = "SfpI2c",
            description   = "Sff8472",
            simpleDisplay = True,
            advanceUser   = False,
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.add(pr.RemoteVariable(
            name        = 'DiagnosticMonitoringType', 
            description = 'Indicates which type of diagnostic monitoring is implemented (if any) in the transceiver',
            offset      = (92 << 2),
            bitSize     = 8, 
            mode        = 'RO',
            base        = pr.UInt,          
        )) 

        self.add(pr.RemoteVariable(   
            name         = 'ExtCalConstantsBlock',
            description  = 'Diagnostic Calibration Constants for Ext Cal',
            offset       = ((256+56) << 2),
            bitSize      = 32*36,
            mode         = 'RO',
            base         = pr.UInt,
        ))           

        self.add(pr.RemoteVariable(   
            name         = 'DiagnosticsBlock',
            description  = 'Diagnostic Monitor Data (internal or external)',
            offset       = ((256+96) << 2),
            bitSize      = 32*10,
            mode         = 'RO',
            base         = pr.UInt,
        ))              

    def get_pwr(self):

        def word(a,o):
            return (a >> (32*o))&0xff
        def tou16(a,o):
            return struct.unpack('H',struct.pack('BB',word(a,o+1),word(a,o)))[0]
        def toi16(a,o):
            return struct.unpack('h',struct.pack('BB',word[a,o+1],word[a,o]))[0]
        def tof32(a,o):
            return struct.unpack('f',struct.pack('BBBB',word(a,o+3),word(a,o+2),word(a,o+1),word(a,o)))[0]

        diag = self.DiagnosticsBlock.get()
        txp = tou16(diag,6)
        rxp = tou16(diag,8)
        rxp_c = [0.]*5
        dtype = self.DiagnosticMonitoringType.get()
        if dtype&(1<<4):
            # External calibration
            calib = self.ExtCalConstantsBlock.get()
            txp_slo  = tou16(calib,24) / 256.
            txp_off  = toi16(calib,26)
            rxp_c[4] = tof32(calib, 0)
            rxp_c[3] = tof32(calib, 4)
            rxp_c[2] = tof32(calib, 8)
            rxp_c[1] = tof32(calib,12)
            rxp_c[0] = tof32(calib,16)
        else:
            txp_slo  = 1.
            txp_off  = 0
            rxp_c[1] = 1.
        txpwr = (txp*txp_slo + txp_off) * 0.0001  # mW
        rxpwr = rxp_c[0]
        rxpn  = rxp
        for i in range(1,5):
            rxpwr += rxp_c[i]*rxpn
            rxpn  *= rxp
        rxpwr *= 0.0001 # mW

        return (txpwr,rxpwr)
