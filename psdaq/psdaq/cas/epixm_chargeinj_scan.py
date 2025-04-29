from psdaq.cas.config_scan_base import ConfigScanBase
from psdaq.configdb.epixm320_config import gain_mode_value
import numpy as np
import json
import logging

nAsics = 4
nColumns = 384

def main():

    # default command line arguments
    defargs = {'--events'  :1024,
               '--hutch'   :'rix',
               '--detname' :'epixm_0',
               '--scantype':'chargeinj',
               '--record'  :1}

    aargs = [('--pulserStep', {'type':int,'default':1,'help':'charge injection ramp step (default 1)'}),
             ('--firstCol',   {'type':int,'default':0,'help':'first column of injection band (default 0)'}),
             ('--lastCol',    {'type':int,'default':nColumns-1,'help':f'last column of injection band (default {nColumns-1})'}),
             ('--bandStep',   {'type':int,'default':0,'help':'number of columns to step injection band by (default 0)'}),
             ('--nBandSteps', {'type':int,'default':1,'help':'number of times to step the injection band (default 1)'}),
             ('--gain-modes', {'type':str,'nargs':'+','choices':['SH','SL','AHL','User'],
                               'default':['SH', 'SL', 'AHL'],
                               'help':'Gain modes to use (default [\'SH\',\'SL\',\'AHL\'])'}),
             ('--asics',      {'type':int,'nargs':'+','choices':range(nAsics),
                               'default':range(nAsics),
                               'help':f'ASICs to use (default {range(nAsics)})'})]
    scan = ConfigScanBase(userargs=aargs, defargs=defargs)

    args = scan.args
    args.pulserStep = max(1, min(1023, args.pulserStep))
    args.firstCol = max(0, min(nColumns-1, args.firstCol))
    args.lastCol  = max(0, min(nColumns-1, args.lastCol))
    if args.firstCol > args.lastCol:
        tmp = args.firstCol
        args.firstCol = args.lastCol
        args.lastCol  = tmp
    # args.bandStep may be:
    # - positive to step the band across the ASIC from left to right
    # - negitive to step the band across the ASIC from right to left
    # - zero to keep the band in one location
    if args.nBandSteps < 1:
        args.nBandSteps = 1

    events = 1024//args.pulserStep # The 1024 is hardcoded in firmware
    if events != args.events:
        logging.warning(f"Overriding to firmware required 1024/{args.pulserStep} = {events} events per step")
        args.events = events

    keys = []
    keys.append(f'{args.detname}:user.gain_mode')
    keys.append(f'{args.detname}:expert.App.FPGAChargeInjection.startCol')
    keys.append(f'{args.detname}:expert.App.FPGAChargeInjection.endCol')
    keys.append(f'{args.detname}:expert.App.FPGAChargeInjection.step')
    keys.append(f'{args.detname}:expert.App.FPGAChargeInjection.currentAsic')

    def steps():
        # The metad goes into the step_docstring of the timing DRP's BeginStep data
        # The step_docstring is used to guide the offline calibration routine
        metad = {'detname'    : args.detname,
                 'scantype'   : args.scantype,
                 'events'     : args.events,
                 'pulserStep' : args.pulserStep,
                 'bandStep'   : args.bandStep,
                 'nBandSteps' : args.nBandSteps}
        d = {}
        d[f'{args.detname}:expert.App.FPGAChargeInjection.step'] = args.pulserStep
        step = 0
        for gain_mode in args.gain_modes:
            metad['gain_mode'] = gain_mode
            d[f'{args.detname}:user.gain_mode'] = int(gain_mode_value(gain_mode))
            for asic in args.asics:
                metad['asic'] = asic
                d[f'{args.detname}:expert.App.FPGAChargeInjection.currentAsic'] = asic
                firstCol = args.firstCol
                lastCol  = args.lastCol
                for bandStep in range(args.nBandSteps):
                    metad['startCol'] = firstCol
                    metad['lastCol']  = lastCol
                    d[f'{args.detname}:expert.App.FPGAChargeInjection.startCol'] = firstCol
                    d[f'{args.detname}:expert.App.FPGAChargeInjection.endCol']   = lastCol

                    metad['step'] = step
                    yield (d, float(step), json.dumps(metad))
                    step += 1

                    firstCol = max(0, min(firstCol + args.bandStep, nColumns-1))
                    lastCol  = max(0, min(lastCol  + args.bandStep, nColumns-1))
                    if lastCol < 0 or firstCol > nColumns-1:  break

    scan.run(keys,steps)

if __name__ == '__main__':
    main()
