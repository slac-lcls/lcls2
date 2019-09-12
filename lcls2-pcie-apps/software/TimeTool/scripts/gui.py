#!/usr/bin/env python3
import setupLibPaths
import pyrogue.gui
import TimeToolDev
import sys
import argparse

#################################################################

# Set the argument parser
parser = argparse.ArgumentParser()

# Convert str to bool
argBool = lambda s: s.lower() in ['true', 't', 'yes', '1']

def auto_int(x):
    return int (x,0)

# Add arguments
parser.add_argument(
    "--dev", 
    type     = str,
    required = False,
    default  = '/dev/datadev_0',
    help     = "path to device",
)  

parser.add_argument(
    "--version3", 
    type     = argBool,
    required = False,
    default  = False,
    help     = "true = PGPv3, false = PGP2b",
) 

parser.add_argument(
    "--pollEn", 
    type     = argBool,
    required = False,
    default  = True,
    help     = "Enable auto-polling",
) 

parser.add_argument(
    "--initRead", 
    type     = argBool,
    required = False,
    default  = True,
    help     = "Enable read all variables at start",
)  

parser.add_argument(
    "--dataDebug", 
    type     = argBool,
    required = False,
    default  = True,
    help     = "Enable TimeToolRx module",
)
parser.add_argument(
    "--enVcMask", 
    type     = auto_int,
    required = False,
    default  = 0xff,
    help     = "mask for pgp read",
)
  

# Get the arguments
args = parser.parse_args()

#################################################################

cl = TimeToolDev.TimeToolDev(
    dev       = args.dev,
    dataDebug = args.dataDebug,
    version3  = args.version3,
    pollEn    = args.pollEn,
    initRead  = args.initRead,
    enVcMask  = args.enVcMask,
)

#################################################################

# Create GUI
appTop = pyrogue.gui.application(sys.argv)
guiTop = pyrogue.gui.GuiTop(group='TimeToolDev')
guiTop.addTree(cl)
guiTop.resize(800, 1000)

# Run gui
appTop.exec_()
cl.stop()

