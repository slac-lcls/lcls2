"""
Shared utilities for Wave8 and Wave8HE configuration.

This module contains common functions and schema helpers used by both
wave8_config.py and wave8he_config.py to avoid code duplication.

Note: importing :mod:`psdaq.utils.enable_lcls2_pgp_pcie_apps` has the
side-effect of adding the axipcie/surf/l2si-core/lcls-timing-core/rogue
submodule paths to ``sys.path`` (via ``pyrogue.addLibraryPath``). This must
run *before* :mod:`psdaq.cas.pgpmonitor` (which does ``import axipcie``),
hence the ordering of the imports below.
"""

import epics
import json
import time
import logging

from psdaq.utils import enable_lcls2_pgp_pcie_apps  # noqa: F401 -- side-effect: sys.path setup for rogue submodules
from psdaq.cas.pgpmonitor import PgpMonitor


# =============================================================================
# EPICS Utilities
# =============================================================================

def ctxt_get(names):
    """Get values from EPICS PVs."""
    v = None
    if isinstance(names, str):
        v = epics.PV(names).get()
    else:
        if isinstance(names, list):
            v = []
            for i, n in enumerate(names):
                v.append(epics.PV(n).get())
    return v


def ctxt_put(names, values):
    """Put values to EPICS PVs."""
    r = []
    print(f'ctxt_put [{names}] [{values}]')
    if isinstance(names, str):
        r.append(epics.PV(names).put(values))
    else:
        if isinstance(names, list):
            for i, n in enumerate(names):
                r.append(epics.PV(n).put(values[i]))
    print(f'returned {r}')


def confirm_xpm_rxid(txId, xpmId, json_str):
    """Verify XPM connection information."""
    json_msg = json.loads(json_str)
    xpm_base = json_msg['body']['control']['0']['control_info']['pv_base']
    xpm_pv = f'{xpm_base}:XPM:{(xpmId>>16)&0xff}:RemoteLinkId{xpmId&0xf}'
    xvalues = int(ctxt_get(xpm_pv))
    if xvalues != txId:
        logging.warning(f'Found 0x{xvalues:x} from {xpm_pv}.  Expected 0x{txId:x}')


def config_timing(epics_prefix, timebase='186M'):
    """Configure LCLS2 timing system."""
    names = [epics_prefix+':Top:SystemRegs:timingUseMiniTpg',
             epics_prefix+':Top:TimingFrameRx:ModeSelEn',
             epics_prefix+':Top:TimingFrameRx:ModeSel',
             epics_prefix+':Top:TimingFrameRx:ClkSel',
             epics_prefix+':Top:TimingFrameRx:RxPllReset']
    values = [0, 1, 1, 1 if timebase=='186M' else 0, 1]
    ctxt_put(names, values)

    time.sleep(1.0)

    names = [epics_prefix+':Top:TimingFrameRx:RxPllReset']
    values = [0]
    ctxt_put(names, values)

    time.sleep(1.0)

    names = [epics_prefix+':Top:TimingFrameRx:RxDown',
             epics_prefix+':Timing:TriggerSource']  # 0=XPM/DAQ, 1=EVR
    values = [0, 0]
    ctxt_put(names, values)


def retrieve_config_from_epics(epics_prefix, scfg, epics_get_func):
    """
    Retrieve full configuration from EPICS PVs for recording.

    Args:
        epics_prefix: The EPICS prefix including ':Top:'
        scfg: The schema configuration dict to populate
        epics_get_func: The epics_get function to use (detector-specific)

    Returns:
        The populated scfg dict
    """
    d = epics_get_func(scfg['expert'])
    keys = [key for key, v in d.items()]
    names = [epics_prefix + v for key, v in d.items()]
    values = ctxt_get(names)
    for i, v in enumerate(values):
        k = keys[i].split('.')
        c = scfg['expert']
        while len(k) > 1:
            c = c[k[0]]
            del k[0]
        if k[0][0] == '[':
            elem = int(k[0][1:-1])
            c[elem] = v if v else c[elem]
        else:
            c[k[0]] = v if v else c[k[0]]
    return scfg


# =============================================================================
# Trigger Delay Configuration (shared by Wave8 and Wave8HE)
# =============================================================================

def configure_trigger_delay(prefix, group, timebase):
    """
    Configure the DAQ TriggerDelay for the current readout group.

    Two modes are supported, distinguished by whether the controls PV
    ``TriggerEventManager:EvrV2CoreTriggers:EvrV2TriggerReg[0]:Delay`` exists:

    * **Old IOC (<3.2.0)** — that PV exists. The script computes
      ``triggerDelay = ctrlDelay - partitionDelay * clksPerFid`` and writes
      it to ``TriggerEventBuffer[0]:TriggerDelay``. Raises ``ValueError`` if
      the result is negative.
    * **New IOC (>=3.2.0)** — the PV no longer exists; the IOC owns
      ``TriggerEventBuffer[0]:TriggerDelay``. The script reads it back and
      raises ``ValueError`` if it is ``None`` or ``0`` (a value of 0 means
      the controls trigger delay is too small, and no triggers will flow).

    Args:
        prefix: EPICS prefix including trailing ``:Top:``.
        group: Readout group index (used to select the XPM partition delay).
        timebase: ``'186M'`` or ``'119M'`` (LCLS-II or LCLS-I derived).
    """
    try:
        # This register no longer exists for IOC firmware/software starting
        # at version 3.2.0. In that mode the IOC manages the delay register
        # (TriggerEventManager:TriggerEventBuffer[0]:TriggerDelay), which is
        # used in both LocalConfig ("standalone") and ReadoutGroup ("daq")
        # modes. - cpo aug 3 2026
        ctrlDelay = ctxt_get(prefix + 'TriggerEventManager:EvrV2CoreTriggers:EvrV2TriggerReg[0]:Delay')
        if ctrlDelay is None:
            print("Failed to retrieve controls trigger delay: IOC controls delay.")
            delayFlag = False
        else:
            delayFlag = True
        partitionDelay = ctxt_get(prefix + 'TriggerEventManager:XpmMessageAligner:PartitionDelay[%d]' % group)

        clksPerFid = 200 if timebase == '186M' else 238
        nsPerClk   = 7000/1300. if timebase == '186M' else 1000/119.

        # Skip if we have IOC software >= 3.2.0; the IOC manages the delay register. - cpo
        if delayFlag:
            # LCLS2 timing. Let controls set the delay value.
            print('ctrlDelay {:}  partitionDelay {:}'.format(ctrlDelay, partitionDelay))

            # Since controls now also runs off the LCLS2 timing fiber, there
            # is no reason to have a "delta". This was put in place to
            # compensate for different LCLS1/LCLS2 timing fiber lengths
            # when controls used the LCLS1 timing fiber. - cpo 02/01/24
            triggerDelay = int(ctrlDelay - partitionDelay * clksPerFid)

            print('triggerDelay {:}'.format(triggerDelay))
            if triggerDelay < 0:
                print('Raise controls trigger delay >= {:} nanoseconds ({:} clock ticks)'.format(
                    -triggerDelay * nsPerClk, -triggerDelay))
                raise ValueError('triggerDelay computes to < 0')

            ctxt_put(prefix + 'TriggerEventManager:TriggerEventBuffer[0]:TriggerDelay', triggerDelay)
        else:
            # New mode where the IOC controls the delay value. The IOC will
            # set this value to 0 if TriggerDelay(ns) is smaller than
            # partitionDelay (a.k.a. L0Delay). Check we have a legal value
            # in the readoutGroup mode that daq uses.
            iocDelay = ctxt_get(prefix + 'TriggerEventManager:TriggerEventBuffer[0]:TriggerDelay')
            if iocDelay is None:
                raise ValueError('Failed to retrieve IOC delay value')
            if iocDelay == 0:
                print('Raise controls trigger delay >= {:} nanoseconds'.format(
                    partitionDelay * nsPerClk))
                raise ValueError('TriggerDelay(ns) too small')

    except KeyError:
        pass


# =============================================================================
# PGP Monitor (PCIe lane health)
# =============================================================================

def init_pgp_monitor(base, dev, lanemask, numVc=2):
    """
    Instantiate a :class:`PgpMonitor` for the Wave8/Wave8HE PCIe board and
    stash it in ``base['pcie']`` so ``connectionInfo``/``config``/``unconfig``
    can call :meth:`PgpMonitor.check_lanes` to assert PGP link health at
    each phase.

    Called once during ``*_init``. ``PgpMonitor.__enter__`` starts the
    embedded ZMQ server; there is no matching ``__exit__`` since the monitor
    lives for the lifetime of the DRP process.

    Args:
        base: Detector state dict (mutated in place).
        dev: Datadev path, e.g. ``'/dev/datadev_0'``.
        lanemask: PGP lane mask (single bit for Wave8/Wave8HE).
        numVc: Number of virtual channels per lane.
    """
    pcie_card = PgpMonitor(pollEn=False,
                           initRead=False,
                           dev=dev,
                           lanemask=lanemask,
                           numVc=numVc)
    pcie_card.__enter__()
    pcie_card.init_lanes()
    base['pcie'] = pcie_card


# =============================================================================
# ADC Delay Constants
# =============================================================================

ADC_DELAY_A_LANE = [
    [0x0c, 0x0b, 0x0e, 0x0e, 0x10, 0x10, 0x12, 0x0b],
    [0x0a, 0x08, 0x0c, 0x0b, 0x0d, 0x0c, 0x0b, 0x0c],
    [0x12, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13],
    [0x0d, 0x0c, 0x0d, 0x0b, 0x0a, 0x12, 0x12, 0x13]
]

ADC_DELAY_B_LANE = [
    [0x11, 0x11, 0x12, 0x12, 0x10, 0x11, 0x0b, 0x0b],
    [0x0a, 0x0a, 0x0c, 0x0c, 0x0c, 0x0b, 0x0b, 0x0a],
    [0x14, 0x14, 0x14, 0x14, 0x14, 0x12, 0x10, 0x11],
    [0x13, 0x12, 0x13, 0x12, 0x12, 0x11, 0x12, 0x11]
]


# =============================================================================
# Schema Helper Functions
# =============================================================================

def set_system_regs(top):
    """Configure SystemRegs block (identical for Wave8 and Wave8HE)."""
    top.set("expert.SystemRegs.AvccEn0", 1, 'UINT8')
    top.set("expert.SystemRegs.AvccEn1", 1, 'UINT8')
    top.set("expert.SystemRegs.Ap5V5En", 1, 'UINT8')
    top.set("expert.SystemRegs.Ap5V0En", 1, 'UINT8')
    top.set("expert.SystemRegs.A0p3V3En", 1, 'UINT8')
    top.set("expert.SystemRegs.A1p3V3En", 1, 'UINT8')
    top.set("expert.SystemRegs.Ap1V8En", 1, 'UINT8')
    top.set("expert.SystemRegs.FpgaTmpCritLatch", 0, 'UINT8')
    top.set("expert.SystemRegs.AdcCtrl1", 0, 'UINT8')
    top.set("expert.SystemRegs.AdcCtrl2", 0, 'UINT8')
    top.set("expert.SystemRegs.TrigEn", 0, 'UINT8')
    top.set("expert.SystemRegs.timingRxUserRst", 0, 'UINT8')
    top.set("expert.SystemRegs.timingTxUserRst", 0, 'UINT8')
    top.set("expert.SystemRegs.timingUseMiniTpg", 0, 'UINT8')
    top.set("expert.SystemRegs.TrigSrcSel", 1, 'UINT8')


def set_raw_buffers(top):
    """Configure RawBuffers block (identical for Wave8 and Wave8HE)."""
    top.set("expert.RawBuffers.BuffEn", [0]*8, 'UINT8')
    top.set("expert.RawBuffers.BuffLen", 100, 'UINT32')
    top.set("expert.RawBuffers.FifoPauseThreshold", 100, 'UINT32')
    top.set("expert.RawBuffers.TrigPrescale", 0, 'INT32')


def set_batcher_event_builder(top):
    """Configure BatcherEventBuilder block (identical for Wave8 and Wave8HE)."""
    top.set("expert.BatcherEventBuilder.Bypass", 0, 'UINT8')
    top.set("expert.BatcherEventBuilder.Timeout", 0, 'UINT32')
    top.set("expert.BatcherEventBuilder.Blowoff", 0, 'UINT8')


def set_trigger_event_manager(top):
    """Configure TriggerEventManager block (identical for Wave8 and Wave8HE)."""
    top.set("expert.TriggerEventManager.TriggerEventBuffer.TriggerDelay", 0, 'UINT32')


def set_adc_readout(top):
    """Configure AdcReadout blocks with delay constants (identical for Wave8 and Wave8HE)."""
    for iadc in range(4):
        adc = 'AdcReadout%d' % iadc
        top.set('expert.' + adc + '.DelayAdcALane', ADC_DELAY_A_LANE[iadc], 'UINT8')
        top.set('expert.' + adc + '.DelayAdcBLane', ADC_DELAY_B_LANE[iadc], 'UINT8')
        top.set('expert.' + adc + '.DMode', 3, 'UINT8')
        top.set('expert.' + adc + '.Invert', 0, 'UINT8')
        top.set('expert.' + adc + '.Convert', 3, 'UINT8')


def set_adc_config(top):
    """Configure AdcConfig blocks (identical for Wave8 and Wave8HE)."""
    for iadc in range(4):
        adc = 'AdcConfig%d' % iadc
        zeroregs = [7, 8, 0xb, 0xc, 0xf, 0x10, 0x11, 0x12, 0x12, 0x13, 0x14, 0x16, 0x17, 0x18, 0x20]
        for r in zeroregs:
            top.set('expert.' + adc + '.AdcReg_0x%04X' % r, 0, 'UINT8')
        top.set('expert.' + adc + '.AdcReg_0x0006', 0x80, 'UINT8')
        top.set('expert.' + adc + '.AdcReg_0x000D', 0x6c, 'UINT8')
        top.set('expert.' + adc + '.AdcReg_0x0015', 1, 'UINT8')
        top.set('expert.' + adc + '.AdcReg_0x001F', 0xff, 'UINT8')


def set_adc_pattern_tester(top):
    """Configure AdcPatternTester block (identical for Wave8 and Wave8HE)."""
    top.set('expert.AdcPatternTester.Channel', 0, 'UINT8')
    top.set('expert.AdcPatternTester.Mask', 0, 'UINT8')
    top.set('expert.AdcPatternTester.Pattern', 0, 'UINT8')
    top.set('expert.AdcPatternTester.Samples', 0, 'UINT32')
    top.set('expert.AdcPatternTester.Request', 0, 'UINT8')


def set_firmware_info(top):
    """Set firmware info placeholders (identical for Wave8 and Wave8HE)."""
    top.set("firmwareBuild:RO", "-", 'CHARSTR')
    top.set("firmwareVersion:RO", 0, 'UINT32')


def define_common_enums(top):
    """Define common enums (identical for Wave8 and Wave8HE)."""
    top.define_enum('baselineEnum', {'_%d_samples' % (2**key): key for key in range(1, 8)})
    top.define_enum('quadrantEnum', {'Even': 0, 'Odd': 1})
