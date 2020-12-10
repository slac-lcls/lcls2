#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue XPM Module
#-----------------------------------------------------------------------------
# File       : CuGenerator.py
# Created    : 2019-06-24
#-----------------------------------------------------------------------------
# Description:
# PyRogue XPM  Module
#-----------------------------------------------------------------------------
# This file is part of the rogue software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue        as pr

from ._Si5317 import *

class XpmInhConfig(pr.Device):

    def __init__(   self, 
            name        = "XpmInhConfig", 
            description = "XPM Inhibit configuration", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        self.add(pr.RemoteVariable(    
            name         = "intv",
            description  = "Interval in 929kHz clks",
            offset       =  0x00,
            bitSize      =  12,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "maxAcc",
            description  = "Maximum accepts within interval",
            offset       =  0x01,
            bitSize      =  4,
            bitOffset    =  0x04,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "inhEn",
            description  = "Enable inhibit",
            offset       =  0x03,
            bitSize      =  1,
            bitOffset    =  0x07,
            base         = pr.UInt,
            mode         = "RW",
        ))



class XpmApp(pr.Device):
    def __init__(   self, 
            name        = "XpmApp", 
            description = "XPM Module", 
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##############################
        # Variables
        ##############################

        self.add(pr.RemoteVariable(    
            name         = "paddr",
            description  = "Timing link address",
            offset       =  0x00,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "partition",
            description  = "partition index",
            offset       =  0x04,
            bitSize      =  4,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "link",
            description  = "link index",
            offset       =  0x04,
            bitSize      =  6,
            bitOffset    =  0x04,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "linkDebug",
            description  = "debug link index",
            offset       =  0x05,
            bitSize      =  4,
            bitOffset    =  0x02,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "amc",
            description  = "amc index",
            offset       =  0x06,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "inhibit",
            description  = "inhibit index",
            offset       =  0x06,
            bitSize      =  2,
            bitOffset    =  0x04,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "tagStream",
            description  = "Enable tag stream fifo input",
            offset       =  0x07,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "usRxEnable",
            description  = "Upstream receive enable",
            offset       =  0x07,
            bitSize      =  1,
            bitOffset    =  0x1,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "cuRxEnable",
            description  = "Copper receive enable",
            offset       =  0x07,
            bitSize      =  1,
            bitOffset    =  0x02,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "fullMask",
            description  = "Readout group full mask",
            offset       =  0x08,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        if False:
            self.add(pr.RemoteVariable(    
                name         = "rxTimeout",
                description  = "Receive timeout",
                offset       =  0x09,
                bitSize      =  9,
                bitOffset    =  0x01,
                base         = pr.UInt,
                mode         = "RW",
            ))

        self.add(pr.RemoteVariable(    
            name         = "txPllReset",
            description  = "Transmit PLL reset",
            offset       =  0x0a,
            bitSize      =  1,
            bitOffset    =  0x02,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "rxPllReset",
            description  = "Receive PLL reset",
            offset       =  0x0a,
            bitSize      =  1,
            bitOffset    =  0x03,
            base         = pr.UInt,
            mode         = "RW",
        ))

        if False:
            self.add(pr.RemoteVariable(    
                name         = "trigSrc",
                description  = "L1 trigger source id",
                offset       =  0x0b,
                bitSize      =  4,
                bitOffset    =  0x00,
                base         = pr.UInt,
                mode         = "RW",
            ))

        self.add(pr.RemoteVariable(    
            name         = "loopback",
            description  = "Loopback enable",
            offset       =  0x0b,
            bitSize      =  1,
            bitOffset    =  0x04,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "txReset",
            description  = "Tx reset",
            offset       =  0x0b,
            bitSize      =  1,
            bitOffset    =  0x05,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "rxReset",
            description  = "Rx reset",
            offset       =  0x0b,
            bitSize      =  1,
            bitOffset    =  0x06,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "fullEn",
            description  = "Deadtime enable",
            offset       =  0x0b,
            bitSize      =  1,
            bitOffset    =  0x07,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "dsLinkStatus",
            description  = "link status summary",
            offset       =  0x0c,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        def _rxErrCnt(var,read):
            return var.dependencies[0].get(read)&0xffff

        self.add(pr.LinkVariable(    
            name         = "rxErrCnt",
            description  = "Receive error counts",
            linkedGet    = _rxErrCnt,
            dependencies = [self.dsLinkStatus]
        ))

        def _txResetDone(var,read):
            return (var.dependencies[0].get(read)>>16)&1

        self.add(pr.LinkVariable(    
            name         = "txResetDone",
            description  = "Tx reset done",
            linkedGet    = _txResetDone,
            dependencies = [self.dsLinkStatus]
        ))

        def _txReady(var,read):
            return (var.dependencies[0].get(read)>>17)&1

        self.add(pr.LinkVariable(    
            name         = "txReady",
            description  = "Tx ready",
            linkedGet    = _txReady,
            dependencies = [self.dsLinkStatus]
        ))

        def _rxResetDone(var,read):
            return (var.dependencies[0].get(read)>>18)&1

        self.add(pr.LinkVariable(    
            name         = "rxResetDone",
            description  = "Rx reset done",
            linkedGet    = _rxResetDone,
            dependencies = [self.dsLinkStatus]
        ))

        def _rxReady(var,read):
            return (var.dependencies[0].get(read)>>19)&1

        self.add(pr.LinkVariable(    
            name         = "rxReady",
            description  = "Rx ready",
            linkedGet    = _rxReady,
            dependencies = [self.dsLinkStatus]
        ))

        def _rxIsXpm(var,read):
            return (var.dependencies[0].get(read)>>20)&1

        self.add(pr.LinkVariable(    
            name         = "rxIsXpm",
            description  = "Remote link partner is XPM",
            linkedGet    = _rxIsXpm,
            dependencies = [self.dsLinkStatus]
        ))

        self.add(pr.RemoteVariable(    
            name         = "dsLinkRxCnt",
            description  = "Rx message count",
            offset       =  0x10,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(Si5317(    
            name         = "amcPLL",
            offset       =  0x14,
        ))

        self.add(pr.RemoteVariable(    
            name         = "l0Reset",
            description  = "L0 trigger reset",
            offset       =  0x18,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "l0En",
            description  = "L0 trigger enable",
            offset       =  0x1a,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "l0Master",
            description  = "L0 trigger master",
            offset       =  0x1b,
            bitSize      =  1,
            bitOffset    =  0x06,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "l0CntEn",
            description  = "L0 counter enable",
            offset       =  0x1b,
            bitSize      =  1,
            bitOffset    =  0x07,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "l0RateSel",
            description  = "L0 rate select",
            offset       =  0x1c,
            bitSize      =  16,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "l0DestSel",
            description  = "L0 destination select",
            offset       =  0x1c,
            bitSize      =  16,
            bitOffset    =  0x10,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "l0Stats",
            description  = "L0 Statistics",
            offset       =  0x20,
            bitSize      =  320,
            bitOffset    =  0x00,
            base         = pr.UInt,
            minimum      =  0.,
            maximum      =  1.,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "numL1Acc",
            description  = "L1 accept count",
            offset       =  0x48,
            bitSize      =  64,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        if False:
            self.add(pr.RemoteVariable(    
                name         = "l1Clr",
                description  = "L1 trigger clear mask",
                offset       =  0x50,
                bitSize      =  16,
                bitOffset    =  0x00,
                base         = pr.UInt,
                mode         = "RW",
            ))

            self.add(pr.RemoteVariable(    
                name         = "l1Ena",
                description  = "L1 trigger enable mask",
                offset       =  0x52,
                bitSize      =  16,
                bitOffset    =  0x00,
                base         = pr.UInt,
                mode         = "RW",
            ))

            self.add(pr.RemoteVariable(    
                name         = "l1Src",
                description  = "L1 trigger source",
                offset       =  0x54,
                bitSize      =  4,
                bitOffset    =  0x00,
                base         = pr.UInt,
                mode         = "RW",
            ))

            self.add(pr.RemoteVariable(    
                name         = "l1Word",
                description  = "L1 trigger word",
                offset       =  0x54,
                bitSize      =  9,
                bitOffset    =  0x04,
                base         = pr.UInt,
                mode         = "RW",
            ))

            self.add(pr.RemoteVariable(    
                name         = "l1Write",
                description  = "L1 trigger write mask",
                offset       =  0x56,
                bitSize      =  1,
                bitOffset    =  0x00,
                base         = pr.UInt,
                mode         = "RW",
            ))

            self.add(pr.RemoteVariable(    
                name         = "anaRst",
                description  = "Analysis tag reset",
                offset       =  0x58,
                bitSize      =  4,
                bitOffset    =  0x00,
                base         = pr.UInt,
                mode         = "RW",
            ))

            self.add(pr.RemoteVariable(    
                name         = "anaTag",
                description  = "Analysis tag",
                offset       =  0x5c,
                bitSize      =  32,
                bitOffset    =  0x00,
                base         = pr.UInt,
                mode         = "RW",
            ))

            self.add(pr.RemoteVariable(    
                name         = "anaPush",
                description  = "Analysis tag insert",
                offset       =  0x60,
                bitSize      =  32,
                bitOffset    =  0x00,
                base         = pr.UInt,
                mode         = "RW",
            ))

            self.add(pr.RemoteVariable(    
                name         = "anaTagWr",
                description  = "Analysis tag write counts",
                offset       =  0x64,
                bitSize      =  32,
                bitOffset    =  0x00,
                base         = pr.UInt,
                mode         = "RO",
            ))

#            self.add(pr.RemoteVariable(    
#                name         = "anaTagRd",
#                description  = "Analysis tag read counts",
#                offset       =  0x68,
#                bitSize      =  32,
#                bitOffset    =  0x00,
#                base         = pr.UInt,
#                mode         = "RO",
#            )) 

        self.add(pr.RemoteVariable(    
            name         = "l0HoldReset",
            description  = "Hold L0Select Reset",
            offset       =  0x68,
            bitSize      =  1,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))
            
        self.add(pr.RemoteVariable(    
            name         = "pipelineDepth",
            description  = "Pipeline depth",
            offset       =  0x6c,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        def pipelineGet(var,read):
            value = var.dependencies[0].get(read)
            return value>>16

        def pipelineSet(deps):
            def pipelineSetValue(var, value, write):
                val = ((value*200)&0xffff) | (value<<16)
                print('Setting pipeline reg to 0x{:x}'.format(val))
                deps[0].set(val,write)
            return pipelineSetValue

        self.add(pr.LinkVariable(
            name         = "l0Delay",
            linkedGet    = pipelineGet,
            linkedSet    = pipelineSet([self.pipelineDepth]),
            dependencies = [self.pipelineDepth]
        ))

        self.add(pr.RemoteVariable(    
            name         = "msgHdr",
            description  = "Message header",
            offset       =  0x70,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False
        ))

        self.add(pr.RemoteVariable(    
            name         = "msgIns",
            description  = "Message insert",
            offset       =  0x71,
            bitSize      =  1,
            bitOffset    =  0x07,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False
        ))

        self.add(pr.RemoteVariable(    
            name         = "msgPayl",
            description  = "Message payload",
            offset       =  0x74,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(    
            name         = "remId",
            description  = "Remote link ID",
            offset       =  0x78,
            bitSize      =  32,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RO",
        ))

        for i in range(4):
            self.add(XpmInhConfig(    
                name         = "inh_%d"%i,
                offset       =  0x80+4*i,
            ))

        self.add(pr.RemoteVariable(
            name         = "inhEvCnt",
            description  = "Inhibit event counts by link",
            offset       =  0x90,
            bitSize      =  1024,
            bitOffset    =  0x00,
            base         = pr.UInt,
            minimum      =  0.,
            maximum      =  1.,
            mode         = "RO",
        ))

        for i in range(4):
            self.add(pr.RemoteVariable(    
                name         = "monClk_%d"%i,
                description  = "Monitor clock rate",
                offset       =  0x110+4*i,
                bitSize      =  29,
                bitOffset    =  0x00,
                base         = pr.UInt,
                mode         = "RO",
            ))

        self.add(pr.RemoteVariable(    
            name         = "inhTmCnt",
            description  = "Inhibit time counts by link",
            offset       =  0x120,
            bitSize      =  1024,
            bitOffset    =  0x00,
            base         = pr.UInt,
            minimum      =  0.,
            maximum      =  1.,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(
            name         = "monStreamPeriod",
            description  = "AxiLite clks between stat tmits",
            offset       = 0x1A0,
            bitSize      = 27,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(
            name         = "monStreamEnable",
            description  = "Enable monitor stream",
            offset       = 0x1A0,
            bitSize      = 1,
            bitOffset    = 31,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(
            name         = "monStreamCount",
            description  = "AxiLite clks since transmit",
            offset       = 0x1A4,
            bitSize      = 27,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(
            name         = "monStreamIndex",
            description  = "Word in transmit",
            offset       = 0x1A8,
            bitSize      = 10,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(
            name         = "monStreamBusy",
            description  = "Transmit in progress",
            offset       = 0x1A8,
            bitSize      = 1,
            bitOffset    = 31,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(
            name         = "monStreamId",
            description  = "Packet in progress",
            offset       = 0x1AC,
            bitSize      = 32,
            bitOffset    = 0,
            base         = pr.UInt,
            mode         = "RO",
        ))

        self.add(pr.RemoteVariable(    
            name         = "groupL0Reset",
            description  = "Group L0 reset",
            offset       =  0x200,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "groupL0Enable",
            description  = "Group L0 enable",
            offset       =  0x204,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "groupL0Disable",
            description  = "Group L0 disable",
            offset       =  0x208,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        self.add(pr.RemoteVariable(    
            name         = "groupMsgInsert",
            description  = "Group message insert",
            offset       =  0x20c,
            bitSize      =  8,
            bitOffset    =  0x00,
            base         = pr.UInt,
            mode         = "RW",
            verify       = False,
        ))

        for i in range(8):
            self.add(pr.RemoteVariable(    
                name         = "stepGroup%i"%i,
                description  = "Step control group",
                offset       =  0x210+8*i,
                bitSize      =  8,
                bitOffset    =  0x00,
                base         = pr.UInt,
                mode         = "RW",
                verify       = False,
            ))
            self.add(pr.RemoteVariable(    
                name         = "stepEnd%i"%i,
                description  = "Step end event",
                offset       =  0x214+8*i,
                bitSize      =  32,
                bitOffset    =  0x00,
                base         = pr.UInt,
                mode         = "RW",
                verify       = False,
            ))

    def l0EnaCnt(self, l0Stats):
        return (l0Stats>>0)&((1<<64)-1)

    def l0EnaTime(self, l0Stats):
        FID_PERIOD = 14.e-6/13.
        return self.l0EnaCnt(l0Stats)*FID_PERIOD

    def l0InhCnt(self, l0Stats):
        return (l0Stats>>64)&((1<<64)-1)

    def numL0(self, l0Stats):
        return (l0Stats>>128)&((1<<64)-1)
        
    def numL0Inh(self, l0Stats):
        return (l0Stats>>192)&((1<<64)-1)
        
    def numL0Acc(self, l0Stats):
        return (l0Stats>>256)&((1<<64)-1)
        
