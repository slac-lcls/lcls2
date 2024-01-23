#-----------------------------------------------------------------------------
# This file is part of the 'LCLS2 Common Carrier Core'. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the 'LCLS2 Common Carrier Core', including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue as pr

class Ds125br401Channel(pr.Device):
    def __init__(self, channel, **kwargs):
        super().__init__(**kwargs)
        '''
        '''


class Ds125br401(pr.Device):
    def __init__(self,
                 name        = "Ds125br401_Channel", 
                 description = "DS125BR401 Channel", 
                 **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        ##########################
        # Device generic registers

        self.add(pr.RemoteVariable(
            name         = "RstSMBus",
            description  = "Reset SMBUS",
            offset       =  (0x07*4),
            bitSize      =  1,
            bitOffset    =  5,
            base         =  pr.Bool,
            mode         =  "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "RstRegisters",
            description  = "Reset internal registers",
            offset       =  (0x07*4),
            bitSize      =  1,
            bitOffset    =  6,
            base         =  pr.Bool,
            mode         =  "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "CtrlEn",
            description  = "Enable SMBus control (1: enable setting VOD, DEM and EQ)",
            offset       =  (0x06*4),
            bitSize      =  1,
            bitOffset    =  3,
            base         =  pr.Bool,
            mode         =  "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "Loopback",
            description  = "Loopback selection",
            offset       =  (0x02*4),
            bitSize      =  2,
            bitOffset    =  4,
            base         =  pr.UInt,
            mode         =  "RW",
            enum         = {0: "Use LPBK pin control", 1: "INA_n to OUTB_n loopback", 2: "INB_n to OUTA_n loopback", 3: "Disable loopback and ignore LPBK pin"}
        ))

        self.add(pr.RemoteVariable(
            name         = "PWDNOverride",
            description  = "Allows overriding PWDN pin (0: prevent, 1: allow)",
            offset       =  (0x02*4),
            bitSize      =  1,
            bitOffset    =  0,
            base         =  pr.Bool,
            mode         =  "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "SD_THOverride",
            description  = "Allows overriding Signal detect (0: prevent, 1: allow)",
            offset       =  (0x08*4),
            bitSize      =  1,
            bitOffset    =  6,
            base         =  pr.Bool,
            mode         =  "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "IDLEOverride",
            description  = "Allows overriding IDLE (0: prevent, 1: allow)",
            offset       =  (0x08*4),
            bitSize      =  1,
            bitOffset    =  4,
            base         =  pr.Bool,
            mode         =  "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "RXDETOverride",
            description  = "Allows overriding RXDET (0: prevent, 1: allow)",
            offset       =  (0x08*4),
            bitSize      =  1,
            bitOffset    =  3,
            base         =  pr.Bool,
            mode         =  "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "MODEOverride",
            description  = "Allows overriding MODE (0: prevent, 1: allow)",
            offset       =  (0x08*4),
            bitSize      =  1,
            bitOffset    =  2,
            base         =  pr.Bool,
            mode         =  "RW",
        ))

        #JM: Loopback (reg: 0x02)

        ########################
        # Channels:
        #    CH0 : Channel B (0)
        #    CH1 : Channel B (1)
        #    CH2 : Channel B (2)
        #    CH3 : Channel B (3)
        #    CH4 : Channel A (0)
        #    CH5 : Channel A (1)
        #    CH6 : Channel A (2)
        #    CH7 : Channel A (3)

        #########################
        # Channels

        channels = [
            'ChannelB[0]', 'ChannelB[1]', 'ChannelB[2]', 'ChannelB[3]', 'ChannelA[0]', 'ChannelA[1]', 'ChannelA[2]', 'ChannelA[3]'
        ]

        for channel in range(len(channels)):
            self.add(pr.RemoteVariable(
                name         = "PWDN_ch[{}]".format(channel),
                description  = "Power down - override pwdn pin. (1: enabled, 0: disabled)",
                offset       =  (0x01*4),
                bitSize      =  1,
                bitOffset    =  channel,
                base         =  pr.Bool,
                mode         =  "RW",
                guiGroup     = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "ShortCircuitProt_ch[{}]".format(channel),
                description  = "Short circuit protection (1: enable, 0: disable)",
                offset       =  ((0x10+0x07*channel)*4),
                bitSize      =  1,
                bitOffset    =  7,
                base         =  pr.Bool,
                mode         =  "RW",
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "ModeSelect_ch[{}]".format(channel),
                description  = "Mode selection (1: PCIe Gen-1 or PCIe Gen-2, 0: PCIe Gen 3)",
                offset       =  ((0x10+0x07*channel)*4),
                bitSize      =  1,
                bitOffset    =  6,
                base         =  pr.Bool,
                mode         =  "RW",
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "SigDet_ch[{}]".format(channel),
                description  = "Signal detect at input (1: not detected, 0: detected)",
                offset       =  (0x0a*4),
                bitSize      =  1,
                bitOffset    =  channel,
                base         =  pr.Bool,
                mode         =  "RO",
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "SigDetRst_ch[{}]".format(channel),
                description  = "Signal detect reset (1: force to OFF)",
                offset       =  ((0x0d+0x07*channel)*4),
                bitSize      =  1,
                bitOffset    =  2,
                base         =  pr.Bool,
                mode         =  "RW",
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "SigDetPreset_ch[{}]".format(channel),
                description  = "Signal detect presset (1: force to ON)",
                offset       =  ((0x0d+0x07*channel)*4),
                bitSize      =  1,
                bitOffset    =  1,
                base         =  pr.Bool,
                mode         =  "RW",
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "IdleSelect_ch[{}]".format(channel),
                description  = "Enable automatic idle detection (0: auto, 1: manual)",
                offset       =  ((0x0e+0x07*channel)*4),
                bitSize      =  1,
                bitOffset    =  5,
                base         =  pr.Bool,
                mode         =  "RW",
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "Idle_ch[{}]".format(channel),
                description  = "Set idle manually (0: output ON, 1: mute output)",
                offset       =  ((0x0e+0x07*channel)*4),
                bitSize      =  1,
                bitOffset    =  4,
                base         =  pr.Bool,
                mode         =  "RW",
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "RxDetConf_ch[{}]".format(channel),
                description  = "Loopback selection",
                offset       =  ((0x0e+0x07*channel)*4),
                bitSize      =  2,
                bitOffset    =  2,
                base         =  pr.UInt,
                mode         =  "RW",
                enum         = {
                        0: "Input is hi-z impedance",
                        1: "Auto RX-Detect for 600ms",
                        2: "Auto RX-Detect until detected",
                        3: "Input is 50 Ω"
                    },
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "EQ_ch[{}]".format(channel),
                description  = "Set equalizer",
                offset       =  ((0x0f+0x07*channel)*4),
                bitSize      =  8,
                bitOffset    =  0,
                base         =  pr.UInt,
                mode         =  "RW",
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "VOD_ch[{}]".format(channel),
                description  = "Set VOD",
                offset       =  ((0x10+0x07*channel)*4),
                bitSize      =  3,
                bitOffset    =  0,
                base         =  pr.UInt,
                mode         =  "RW",
                enum         = {
                        0: "0.7 V",
                        1: "0.8 V",
                        2: "0.9 V",
                        3: "1 V",
                        4: "1.1 V",
                        5: "1.2 V (default)",
                        6: "1.3 V",
                        7: "1.4 V"
                    },
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "Rxdet_ch[{}]".format(channel),
                description  = "Rx detected (1: detected)",
                offset       =  ((0x11+0x07*channel)*4),
                bitSize      =  1,
                bitOffset    =  7,
                base         =  pr.Bool,
                mode         =  "RO",
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "Mode_ch[{}]".format(channel),
                description  = "Mode",
                offset       =  ((0x11+0x07*channel)*4),
                bitSize      =  2,
                bitOffset    =  5,
                base         =  pr.UInt,
                mode         =  "RO",
                enum         = {
                        0: "PCIe Gen-1 (2.5G)",
                        1: "PCIe Gen-2 (5G)",
                        2: "Not defined",
                        3: "PCIe Gen-3 (8G+)"
                    },
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "DEM_ch[{}]".format(channel),
                description  = "DEM Control",
                offset       =  ((0x11+0x07*channel)*4),
                bitSize      =  3,
                bitOffset    =  0,
                base         =  pr.UInt,
                mode         =  "RW",
                enum         = {
                        0: "0 dB",
                        1: "-1.5 dB",
                        2: "-3.5 dB (default)",
                        3: "-5 dB",
                        4: "-6 dB",
                        5: "-8 dB",
                        6: "-9 dB",
                        7: "-12 dB"
                    },
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "IdleAssertThreshold_ch[{}]".format(channel),
                description  = "Idle assert threshold",
                offset       =  ((0x12+0x07*channel)*4),
                bitSize      =  2,
                bitOffset    =  2,
                base         =  pr.UInt,
                mode         =  "RW",
                enum         = {
                        0: "180 mVp-p (default)",
                        1: "160 mVp-p",
                        2: "210 mVp-p",
                        3: "190 mVp-p"
                    },
                guiGroup    = '{}'.format(channels[channel])
            ))

            self.add(pr.RemoteVariable(
                name         = "IdleDeassertThreshold_ch[{}]".format(channel),
                description  = "Idle deassert threshold",
                offset       =  ((0x12+0x07*channel)*4),
                bitSize      =  2,
                bitOffset    =  0,
                base         =  pr.UInt,
                mode         =  "RW",
                enum         = {
                        0: "110 mVp-p (default)",
                        1: "100 mVp-p",
                        2: "150 mVp-p",
                        3: "130 mVp-p"
                    },
                guiGroup    = '{}'.format(channels[channel])
            ))

        @self.command(name="Init", description="Initialize bus",)
        def Init():
            print("INIT Ds125br401 (nothing to do yet)")
