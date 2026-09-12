"""Extra timing set-up for the ePixUHR emulator firmware.

The emulator needs LCLS-II timing configured explicitly, which the real detectors
do not, and which is otherwise done by hand from the devGui by clicking
TDetTiming.ConfigLclsTimingV2.  That is tedious on a multi-datadev node, so this
does it from the DRP.

Everything else is xpmdet_config's job.  This deliberately does not duplicate or
wrap any of it: it reaches the rogue tree through xpmdet_config's own module
global, so xpmdet_config and Drp::XpmDetector both stay untouched and the CPU DRPs
are unaffected.

Called from the Gpu::EpixUHRemu side of the GPU DRP, before
xpmdet_connectionInfo() runs.  It has to be before: with the link down,
xpmdet_connectionInfo() reads the XPM remote link id as 0xffffffff and raises
'Illegal XPM Remote link id', so anything hooked after it never executes.  Its own
RxPllReset retry is not enough, because ConfigLclsTimingV2() also clears UseMiniTpg
and issues TxPhyReset and the Tx and Rx user resets.
"""

import logging
import time

from psdaq.configdb import xpmdet_config


def epixuhremu_configTiming():
    """Configure LCLS-II timing if, and only if, the link is not already up.

    ConfigLclsTimingV2() is not free or side-effect-free: it resets the receive
    PLL, issues Tx and Rx user resets, and sleeps three times for a second, so
    calling it unconditionally would add that to every Allocate and bounce a link
    that was working.  Hence the guard.

    RxLinkUp is the live link status; RxDown is a latch that records that the link
    went down at some point.  So the decision is made on RxLinkUp, and the latch is
    cleared afterwards, as the rest of the timing code does.

    Note that this runs in every DRP process, not just the barrier supervisor that
    xpmdet_connectionInfo() elects, because it has to happen before that election.
    Each process configures the card it opened, so that is right where there is one
    process per card, as for the emulator.  Were two processes ever to share a card,
    the RxLinkUp guard would make the second a no-op in the common case but would not
    make it safe: they could both find the link down and reset it in turn.
    """
    root = xpmdet_config.args['root']
    tim  = root.TDetTiming.TimingFrameRx

    if tim.RxLinkUp.get():
        logging.info('epixuhremu: timing link is up, leaving it alone')
        return

    logging.warning('epixuhremu: timing link is down, calling ConfigLclsTimingV2()')
    root.TDetTiming.ConfigLclsTimingV2()

    tim.RxDown.set(0)                   # Reset the latching register

    # ConfigLclsTimingV2 already sleeps, but the link needs a moment after the
    # last reset before RxLinkUp means anything.
    time.sleep(1.0)
    if tim.RxLinkUp.get():
        logging.info('epixuhremu: timing link is up')
    else:
        logging.error('epixuhremu: timing link is still down after '
                      'ConfigLclsTimingV2(); check the fibre and the XPM')
