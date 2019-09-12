#!/usr/bin/env python3
import setupLibPaths
import TimeToolDev
import sys
import time
import gc
import IPython
import argparse


def toggle_acquisition(x):

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


    cl.StartRun()
    time.sleep(x)
    cl.StopRun()


    cl.stop()

if __name__ == "__main__":
    toggle_acquisition(5)
