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


def _dumpPhy(root):
    """Transmit and receive PHY status, one line, at a level a DRP actually prints.

    Cheap: four register reads once per Allocate.  Worth it because the interesting
    failure is asymmetric -- the receive link can be up and locked while the transmitter
    is not reaching the XPM -- and nothing else in the log distinguishes the two
    directions.  TxClkFreq near zero, or a TxRstStatus that never clears, says the
    transmit side is the problem and that ConfigLclsTimingV2 (which issues TxPhyReset and
    TxUserRst) is worth forcing even though RxLinkUp looks fine.

    Loopback is here because a *far-end* mode is invisible from this side in every other
    respect: the receive link stays healthy and RxId reads correctly, while the card
    retransmits what it receives instead of its own TxId, so the XPM sees garbage rather
    than TDetSim/<host> and counts errors without ever landing a valid frame.  A near-end
    mode is self-evident by contrast -- RxId could not decode -- so it is the far-end case
    that needs stating.  UseMiniTpg for the same reason: it changes where timing comes from
    and nothing else in the log records it.
    """
    try:
        phy = root.TDetTiming.TimingPhyMonitor
        tim = root.TDetTiming.TimingFrameRx
        logging.warning('epixuhremu: RxLinkUp %s  MmcmLocked %s  '
                        'TxRstStatus 0x%x  RxRstStatus 0x%x  '
                        'TxClkFreq %.3f MHz  RxClkFreq %.3f MHz  '
                        'Loopback %s  UseMiniTpg %s',
                        tim.RxLinkUp.get(), phy.MmcmLocked.get(),
                        phy.TxRstStatus.get(), phy.RxRstStatus.get(),
                        phy.TxClkFreq.get() * 1.e-6, phy.RxClkFreq.get() * 1.e-6,
                        phy.Loopback.getDisp(), phy.UseMiniTpg.get())
    except Exception as exc:
        # Diagnostics must never be the reason Allocate fails.
        logging.warning('epixuhremu: could not read PHY status: %s', exc)

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

    # Both directions, before deciding anything.  RxLinkUp describes only the receive
    # side, and on drp-srcf-gpu008 on 2026-09-15 a card with a healthy receive link was
    # not being seen by the XPM at all -- xpmpva showed RemoteLinkId as undef/0 where it
    # should have shown TDetSim/gpu008 -- so the fault was on the transmit side, which the
    # guard below cannot see.  Log the transmit status too, so that case is visible
    # instead of having to be inferred from which messages are missing.
    _dumpPhy(root)

    if tim.RxLinkUp.get():
        # warning, not info: the surrounding timing code logs at WARNING, so INFO is
        # filtered out in a DRP and this decision would leave no trace at all.  Both
        # branches have to be visible or a log cannot distinguish "ran and skipped" from
        # "never ran".  It is one line per Allocate, so there is no noise cost.
        logging.warning('epixuhremu: timing link is up, leaving it alone')
        return

    logging.warning('epixuhremu: timing link is down, calling ConfigLclsTimingV2()')
    root.TDetTiming.ConfigLclsTimingV2()

    tim.RxDown.set(0)                   # Reset the latching register

    # ConfigLclsTimingV2 already sleeps, but the link needs a moment after the
    # last reset before RxLinkUp means anything.
    time.sleep(1.0)
    if tim.RxLinkUp.get():
        # Everything the counters hold was accumulated while the link was training,
        # so it says nothing about the run about to start.  Clearing here matters
        # because xpmdet_connectionInfo() dumps them a moment later and only then
        # clears them itself, so without this the numbers in the log are noise.
        tim.ClearRxCounters()
        logging.warning('epixuhremu: timing link is up, Rx counters cleared')
    else:
        # Left uncleared deliberately: with the link still down, the error counts are
        # the evidence.
        logging.error('epixuhremu: timing link is still down after '
                      'ConfigLclsTimingV2(); check the fibre and the XPM')
