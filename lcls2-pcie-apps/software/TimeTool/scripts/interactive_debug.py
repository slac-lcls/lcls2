#!/usr/bin/env python3
import setupLibPaths
import TimeToolDev
import sys
import time
import gc
 
cl = TimeToolDev.TimeToolDev(True)

def frame_rate(sec):
    start_time  = time.time()
    start_count = cl.ClinkTest.ClinkTop.ChannelA.FrameCount.get()

    time.sleep(sec)

    stop_time  = time.time()
    stop_count = cl.ClinkTest.ClinkTop.ChannelA.FrameCount.get()
