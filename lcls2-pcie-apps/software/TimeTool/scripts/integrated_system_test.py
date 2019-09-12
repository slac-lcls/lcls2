#!/usr/bin/env python3
import setupLibPaths
import TimeToolDev
import sys
import time
import gc
import IPython
import argparse
import numpy as np

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
    dataDebug = True,
    version3  = args.version3,
    pollEn    = False,
    initRead  = False,
)
if(cl.dev[0]=='sim'):
	cl.LoadConfig("config/TimeToolVcsSimTest_lcls-pc823236.yml")
	cl.Application.AppLane[0].EventBuilder.Bypass.set(0x1)
else:
	cl.LoadConfig("config/slow_test_pattern_tpg_trig.yml")
	cl.Application.AppLane[0].EventBuilder.Bypass.set(0x0)
	cl.Hardware.Timing.Triggering.LocalTrig[0].EnableTrig.set(False)

cl.Application.AppLane[0].ByPass.ByPass.set(0x1)


cl.StartRun()

time.sleep(3)

if(cl.dev[0]=='sim'):
    #p =  np.array(cl._frameGen[0].make_byte_array())
    p = (np.random.rand(300)*255).astype(np.uint8)
    cl.GenUserFrame[0]()(p)
else:
    cl.Hardware.Timing.Triggering.LocalTrig[0].EnableTrig.set(True)

cl.Application.AppLane[0].Prescale.DialInPreScaling.set(0)



start_count = 0

my_results_list = []

for i in range(4):
      if(cl.dev[0]!='sim'):
            break
      print("counter = "+str(start_count))

      #p =  np.array(cl._frameGen[0].make_byte_array())
      p = (np.random.rand(300)*255).astype(np.uint8)

      if(cl.dev[0]=='sim'):
            cl.GenUserFrame[0]()(p)

      too_many_counter  = 0
      while(cl.TimeToolRx.frameCount.get()<start_count+1):
            too_many_counter = too_many_counter +1
            time.sleep(1)
            if(too_many_counter>20): break

      start_count = cl.TimeToolRx.frameCount.get()
      my_mask = np.arange(36)

      if(len(p)>100):
            my_mask = np.append(my_mask,np.arange(int(len(p)/2),int(len(p)/2)+36))
            my_mask = np.append(my_mask,np.arange(len(p)-36,len(p)))

      #print(p[my_mask] == cl.TimeToolRx.parsed_data)
      #assert (not False in (p[my_mask] == cl.TimeToolRx.parsed_data))
      #print(p[my_mask])
      print("data_out = "+str(cl.TimeToolRx.parsed_data))

time.sleep(1)

if(cl.dev[0]!='sim'):
    cl.Hardware.Timing.Triggering.LocalTrig[0].EnableTrig.set(False)

cl.stop()
