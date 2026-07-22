"""
Shared utilities for Wave8 and Wave8HE configuration.

This module contains common functions and schema helpers used by both
wave8_config.py and wave8he_config.py to avoid code duplication.
"""

import epics
import json
import time
import logging


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
