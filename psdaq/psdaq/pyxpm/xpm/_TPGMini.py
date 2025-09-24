#-----------------------------------------------------------------------------
# Title      : PyRogue Embedded timing pattern generator
#-----------------------------------------------------------------------------
# Description:
# PyRogue Embedded timing pattern generator
#-----------------------------------------------------------------------------
# This file is part of the 'LCLS Timing Core'. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the 'LCLS Timing Core', including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue as pr

class TPGMini(pr.Device):
    def __init__(   self,
            name        = "TPGMini",
            description = "Embedded timing pattern generator",
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(
            name         = "ClockAdvanceRate",
            description  = "Timestamp fractional divisor",
            offset       = 0x00,
            bitSize      = 32,
            bitOffset    = 0,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "BaseControl",
            description  = "Base rate trigger divisor",
            offset       = 0x04,
            bitSize      = 16,
            bitOffset    = 0,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "PulseIdWr",
            description  = "Pulse ID write",
            offset       = 0x58,
            bitSize      = 64,
            bitOffset    = 0,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(
            name         = "PulseIdRd",
            description  = "Pulse ID read",
            offset       = 0x08,
            bitSize      = 64,
            bitOffset    = 0,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(
            name         = "PulseIdSet",
            description  = "Activates PulseId register value",
            offset       = 0x70,
            bitSize      = 1,
            bitOffset    = 0,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(
            name         = "TStampWr",
            description  = "Time stamp Write",
            offset       = 0x60,
            bitSize      = 64,
            bitOffset    = 0,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(
            name         = "TStampRd",
            description  = "Time stamp read",
            offset       = 0x10,
            bitSize      = 64,
            bitOffset    = 0,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(
            name         = "TStampSet",
            description  = "Activates Timestamp register value",
            offset       = 0x74,
            bitSize      = 1,
            bitOffset    = 0,
            mode         = "WO",
        ))

        self.addRemoteVariables(
            name         = "FixedRateDiv",
            description  = "Fixed rate marker divisors",
            offset       = 0x18,
            bitSize      = 32,
            bitOffset    = 0,
            mode         = "RW",
            number       = 10,
            stride       = 4,
            hidden       = True,
        )

        self.add(pr.RemoteVariable(
            name         = "RateReload",
            description  = "Loads cached fixed rate marker divisors",
            offset       = 0x40,
            bitSize      = 1,
            bitOffset    = 0,
            mode         = "WO",
        ))

        self.addRemoteVariables(
            name         = "ACRateDiv",
            description  = "AC rate marker divisors",
            offset       = 0x80,
            bitSize      = 8,
            bitOffset    = 0,
            mode         = "RW",
            number       = 6,
            stride       = 4,
            hidden       = True,
        )

        self.add(pr.RemoteVariable(
            name         = "ACRateReload",
            description  = "Loads cached AC rate marker divisors",
            offset       = 0x98,
            bitSize      = 1,
            bitOffset    = 0,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(
            name         = "NBeamSeq",
            description  = "Number of beam request engines",
            offset       = 0x4C,
            bitSize      = 6,
            bitOffset    = 0,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(
            name         = "NControlSeq",
            description  = "Number of control sequence engines",
            offset       = 0x4C,
            bitSize      = 8,
            bitOffset    = 6,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(
            name         = "NArraysBsa",
            description  = "Number of BSA arrays",
            offset       = 0x4C,
            bitSize      = 8,
            bitOffset    = 14,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(
            name         = "SeqAddrLen",
            description  = "Number of beam sequence engines",
            offset       = 0x4C,
            bitSize      = 4,
            bitOffset    = 22,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(
            name         = "NAllowSeq",
            description  = "Number of beam allow engines",
            offset       = 0x4C,
            bitSize      = 6,
            bitOffset    = 26,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(
            name         = "TxReset",
            description  = "Reset transmit link",
            offset       = 0x68,
            bitSize      = 1,
            bitOffset    = 0,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(
            name         = "CountIntervalReset",
            description  = "Count Interval Reset",
            offset       = 0x6C,
            bitSize      = 1,
            bitOffset    = 0,
            mode         = "WO",
        ))

        self.add(pr.RemoteVariable(
            name         = "PllCnt",
            description  = "Count of PLL status changes",
            offset       = 0x500,
            bitSize      = 32,
            bitOffset    = 0,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(
            name         = "ClkCnt",
            description  = "Count of local 186M clock",
            offset       = 0x504,
            bitSize      = 32,
            bitOffset    = 0,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(
            name         = "SyncErrCnt",
            description  = "Count of 71k sync errors",
            offset       = 0x508,
            bitSize      = 32,
            bitOffset    = 0,
            mode         = "RO",
            pollInterval = 1,
        ))

        self.add(pr.RemoteVariable(
            name         = "CountInterval",
            description  = "Interval counters update period",
            offset       = 0x50C,
            bitSize      = 32,
            bitOffset    = 0,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "BaseRateCount",
            description  = "Count of base rate triggers",
            offset       = 0x510,
            bitSize      = 32,
            bitOffset    = 0,
            mode         = "RO",
            pollInterval = 1,
        ))

    def setup(self,isUED):
        if True:
            if isUED:
                self.BaseControl.set(238)
                #  Set the fixed rate markers
                self.FixedRateDiv[0].set(500000)
                self.FixedRateDiv[1].set(  1000)
                self.FixedRateDiv[2].set(   500)
                self.FixedRateDiv[3].set(   100)
                self.FixedRateDiv[4].set(    50)
                self.FixedRateDiv[5].set(    10)
                self.FixedRateDiv[6].set(     5)
                #  And some rate markers tied to beam simulation
                self.FixedRateDiv[7].set(     1)  # beam request
                self.FixedRateDiv[8].set(  1000)  # dest b1
                self.FixedRateDiv[9].set(   500)  # dest b2
                self.RateReload.set(1)

            else:
                self.BaseControl.set(200)
                #  Set the fixed rate markers
                self.FixedRateDiv[0].set(910000)
                self.FixedRateDiv[1].set( 91000)
                self.FixedRateDiv[2].set(  9100)
                self.FixedRateDiv[3].set(   910)
                self.FixedRateDiv[4].set(    91)
                self.FixedRateDiv[5].set(    13)
                self.FixedRateDiv[6].set(     1)
                #  And some rate markers tied to beam simulation
                self.FixedRateDiv[7].set(  9100)  # beam request
                self.FixedRateDiv[8].set( 91000)  # dest b1
                self.FixedRateDiv[9].set(  9100)  # dest b2
                self.RateReload.set(1)

            #  Set the AC rate markers
            self.ACRateDiv[0].set(120)
            self.ACRateDiv[1].set( 60)
            self.ACRateDiv[2].set( 12)
            self.ACRateDiv[3].set(  6)
            self.ACRateDiv[4].set(  2)
            self.ACRateDiv[5].set(  1)
            self.ACRateReload.set(1)


