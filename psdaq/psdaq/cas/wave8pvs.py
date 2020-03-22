import rogue.hardware.pgp
import pyrogue
import pyrogue.utilities.prbs
import pyrogue.utilities.fileio
import pyrogue.protocols.epicsV4
import threading
import signal
import atexit
import yaml
import time
import sys
import argparse
import wave8 as w8
#from AdmPcieKu3Pgp2b import *
import pyrogue.utilities.prbs

# Set the argument parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument(
    "--l", 
    type     = int,
    required = False,
    default  = 0,
    help     = "PGP lane number",
)

parser.add_argument(
    "--dev", 
    type     = str,
    required = False,
    default  = '/dev/datadev_0',
    help     = "PGP device (default /dev/datadev_0)",
)  

parser.add_argument(
    "--hvBay0En", 
    type     = bool,
    required = False,
    default  = False,
    help     = "Enable HV generator bay 0",
)

parser.add_argument(
    "--hvBay1En", 
    type     = bool,
    required = False,
    default  = False,
    help     = "Enable HV generator bay 1",
)

parser.add_argument(
    "--base", 
    type     = str,
    required = False,
    default  = 'DAQ:WAVE8',
    help     = "PVA Base",
)

# Get the arguments
args = parser.parse_args()

# Set base
Wave8Board = w8.Top(hwType='datadev', dev=args.dev, lane=args.l, hvBay0En=args.hvBay0En, hvBay1En=args.hvBay1En, dataCapture=False)

epics = pyrogue.protocols.epicsV4.EpicsPvServer(base=args.base,root=Wave8Board,incGroups=None,excGroups=['NoPVA'])

def main():

    # Start the system
    Wave8Board.start(
        pollEn   = True,
        initRead = True,
        timeout  = 5.0,    
    )

    # Start EPICS
    epics.start()
    epics.dump()

# Close window and stop polling
def stop():
    mNode.stop()
    Wave8Board.stop()
    exit()

if __name__ == '__main__':
    main()

    print('Main complete')
