#!/usr/bin/env python3
#import setupLibPaths
import TimeToolDev
#import sys
#import time
#import gc
#import IPython
#import argparse


def toggle_prescaling():

    #################################################################
    cl = TimeToolDev.TimeToolDev(
        dev       = '/dev/datadev_0',
        dataDebug = False,
        version3  = False,
        pollEn    = False,
        initRead  = False,
        enVcMask  = 0xD,
    )

    #################################################################

    if(cl.Hardware.PgpMon[0].RxRemLinkReady.get() != 1):
        raise ValueError(f'PGP Link is down' )
        
    #################################################################


    #cl.StartRun()
    #time.sleep(x)
    #cl.StopRun()

    prescaling = cl.Application.AppLane[0].Prescale.ScratchPad.get()
    print("old prescaler = ",prescaling)
    if(prescaling == 2):
        prescaling = 6
    else:
        prescaling = 2

    
    cl.Application.AppLane[0].Prescale.ScratchPad.set(prescaling)

    print("new prescaler = ",cl.Application.AppLane[0].Prescale.ScratchPad.get())


    cl.stop()

if __name__ == "__main__":
    toggle_prescaling()
