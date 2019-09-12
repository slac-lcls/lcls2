#!/usr/bin/env python3
import setupLibPaths
import TimeToolDev
import sys
import time
import gc

import argparse

#################################################################

# Set the argument parser
parser = argparse.ArgumentParser()

# Convert str to bool
argBool = lambda s: s.lower() in ['true', 't', 'yes', '1']

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

# Get the arguments
args = parser.parse_args()

#################################################################

cl = TimeToolDev.TimeToolDev(
    dev       = args.dev,
    dataDebug = args.dataDebug,
    version3  = args.version3,
    pollEn    = args.pollEn,
    initRead  = args.initRead,
)

#################################################################

if(cl.Hardware.PgpMon[0].RxRemLinkReady.get() != 1):
    raise ValueError(f'PGP Link is down' )
    
#################################################################

#these commands are sent over serial to the camera unit
#spf: 8-bit mode
#clm: Camera Mink Mode. 0 = base, 1 = medium, 2 = full, 3 = deca
#svm: test pattern mode
#sem: Set Exposure Mode
#set: Set Exposure Time
#stm: External Trigger Mode

my_commands_to_pirana_camera = ['spf 0', 'clm 2','svm 0', 'sem 0', 'set 5000', 'stm 1']

for i in my_commands_to_pirana_camera:
	print("sent command: "+i)
	cl.ClinkFeb[0].UartPiranha4[0]._tx.sendString(i)
    
time.sleep(3)
cl.ClinkFeb[0].UartPiranha4[0].GCP()

cl.ClinkFeb[0].ClinkTop.Channel[0].DataEn.get()
cl.ClinkFeb[0].ClinkTop.Channel[0].DataEn.set(True)	#this start the data collection
time.sleep(1)
cl.ClinkFeb[0].ClinkTop.Channel[0].DataEn.set(False)#this stops the data collection


#validating prescalling

cl.ClinkFeb[0].UartPiranha4[0]._tx.sendString('svm 0')    #test pattern. 1 is Ramp, 0 is sensor video
cl.ClinkFeb[0].UartPiranha4[0]._tx.sendString('stm 1')    #1 is internal trigger, 2 is External pulse width. Manual description is off by one

def frame_rate(sec):
    start_time  = time.time()
    start_count = cl.ClinkFeb[0].ClinkTop.Channel[0].FrameCount.get()

    time.sleep(sec)

    stop_time  = time.time()
    stop_count = cl.ClinkFeb[0].ClinkTop.Channel[0].FrameCount.get()

    return (stop_count-start_count)*1.0/(stop_time-start_time)

#cl.stop()	#does this need cl.start() counter part? don't see it in gui.py
#time.sleep(1)
#cl._dbg.close_h5_file()



cl.ClinkFeb[0].UartPiranha4[0]._tx.sendString('ssf 7000')
cl.ClinkFeb[0].UartPiranha4[0]._tx.sendString('ssf 6000')
cl.ClinkFeb[0].ClinkTop.Channel[0].DropCount.get()
cl.Application.AppLane[0].Prescale.DialInPreScaling.set(254)
cl.Application.AppLane[0].Prescale.DialInPreScaling.set(124)
cl.Application.AppLane[0].Prescale.DialInPreScaling.set(125)
cl.ClinkFeb[0].ClinkTop.Channel[0].DropCount.get()
frame_rate(2)

# Close the root class and exit
cl.stop()
exit()
