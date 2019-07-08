#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PyRogue _Si5317 Module
#-----------------------------------------------------------------------------
# File       : _Si5317.py
# Created    : 2017-01-17
# Last update: 2017-01-17
#-----------------------------------------------------------------------------
# Description:
# PyRogue _Si5317 Module
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
import csv
import click

class Si5317(pr.Device):
    def __init__(self,       
            name          = "Si5317",
            description   = "Si5317",
            simpleDisplay = True,
            advanceUser   = False,
            **kwargs):
        super().__init__(name=name, description=description, **kwargs)

        for i in range(2):
            self.add(pr.RemoteVariable(
                name        = 'bwSel%d'%i,
                description = 'Loop filter bandwidth selection',
                offset      = 0,
                bitSize     = 2,
                bitOffset   = 2*i,
                mode        = 'RW',
            ))
            self.add(pr.RemoteVariable(
                name        = 'rate%d'%i,
                description = 'Rate',
                offset      = 2,
                bitSize     = 2,
                bitOffset   = 2*i,
                mode        = 'RW',
            ))        

        self.add(pr.RemoteVariable(
            name        = 'frqTbl',
            description = 'Frequency table {L,H,M}',
            offset      = 0,
            bitSize     = 2,
            bitOffset   = 4,
            mode        = 'RW',
        ))        

        for i in range(4):
            self.add(pr.RemoteVariable(
                name        = 'frqSel%d'%i,
                description = 'Frequency selection',
                offset      = 1,
                bitSize     = 2,
                bitOffset   = 2*i,
                mode        = 'RW',
            ))        

        self.add(pr.RemoteVariable(
            name        = 'inc',
            description = 'Increment phase',
            offset      = 2,
            bitSize     = 1,
            bitOffset   = 4,
            mode        = 'RW',
        ))        

        self.add(pr.RemoteVariable(
            name        = 'dec',
            description = 'Decrement phase',
            offset      = 2,
            bitSize     = 1,
            bitOffset   = 5,
            mode        = 'RW',
        ))        

        self.add(pr.RemoteVariable(
            name        = 'bypass',
            description = 'Bypass',
            offset      = 2,
            bitSize     = 1,
            bitOffset   = 6,
            mode        = 'RW',
        ))        

        self.add(pr.RemoteVariable(
            name        = 'rstn',
            description = 'ResetN',
            offset      = 2,
            bitSize     = 1,
            bitOffset   = 7,
            mode        = 'RW',
        ))        

        self.add(pr.RemoteVariable(
            name        = 'lolCnt',
            description = 'Loss of lock count',
            offset      = 3,
            bitSize     = 3,
            bitOffset   = 0,
            mode        = 'RW',
        ))        

        self.add(pr.RemoteVariable(
            name        = 'lol',
            description = 'Loss of lock status',
            offset      = 3,
            bitSize     = 1,
            bitOffset   = 3,
            mode        = 'RW',
        ))        

        self.add(pr.RemoteVariable(
            name        = 'losCnt',
            description = 'Loss of signal count',
            offset      = 3,
            bitSize     = 3,
            bitOffset   = 4,
            mode        = 'RW',
        ))        

        self.add(pr.RemoteVariable(
            name        = 'los',
            description = 'Loss of signal status',
            offset      = 3,
            bitSize     = 1,
            bitOffset   = 7,
            mode        = 'RW',
        ))        

        @self.command(name='Dump',description='dump all csr')
        def _Dump(self):
            def get(name):
                return self.find(name=name).get()

            lmh = ['L', 'H', 'M', 'm']
            yn = ['N','Y']
            print("  "),
            print("FrqTbl %c  ", lmh[get('frqTbl')]),
            print("FrqSel %c%c%c%c  ",
                  lmh[get('frqSel3')],
                  lmh[get('frqSel2')],
                  lmh[get('frqSel1')],
                  lmh[get('frqSel0')]),
            print("BwSel %c%c  ", lmh[get('bwSel1')], lmh[get('bwSel0')]),
            print("Rate %c%c  ", lmh[get('rate1')], lmh[get('rate0')]),
            print("LOLCnt %u  ", get('lolCnt')),
            print("LOL %c  ", yn[get('lol')]),
            print("LOSCnt %u  ", get('losCnt')),
            print("LOL %c  ", yn[get('los')])
