#!/usr/bin/env python

import sys
import pyrogue as pr
import argparse

import logging

from psdaq.pyxpm.xpm.Top import *

def main():
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='reprogram XPM PROM')

    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    parser.add_argument('--file', type=str, required=True, help="MCS file" )
    parser.add_argument('--ip', type=str, required=True, help="IP address" )

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Set base
    base = pr.Root(name='AMCc',description='') 

    base.add(Top(
        name   = 'XPM',
        ipAddr = args.ip
    ))
    
    # Start the system
    base.start(
        pollEn   = False,
        initRead = False,
        zmqPort  = None,
    )

    AxiVersion = base.XPM.AxiVersion
    PROM       = base.XPM.MicronN25Q

    AxiVersion.printStatus()

    PROM.LoadMcsFile(args.file)

    if(PROM._progDone):
        print('\nReloading FPGA firmware from PROM ....')
        AxiVersion.FpgaReload()
        time.sleep(10)
        print('\nReloading FPGA done')

        print ( '###################################################')
        print ( '#                 New Firmware                    #')
        print ( '###################################################')
        AxiVersion.printStatus()
    else:
        print('Failed to program FPGA')
        
    base.stop()
    exit()

if __name__ == '__main__':
    main()
