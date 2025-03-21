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
               '--record'  :1,
               }

    aargs = [('--pulserStep',  {'type':int,'default':1,'help':'charge injection ramp step (default 1)'}),
             ('--pulserStart', {'type':int,'default':0,'help':'charge injection ramp step (default 0)'}),
             ('--pulserStop',  {'type':int,'default':1,'help':'charge injection ramp step (default 1)'}),
             ('--gain-modes',  {'type':str,
                               'default':'FHGIon, FMGIon, FLG1Ion, FLG2Ion',
                               'help':'Gain modes to use (default [\'FHGIon\',\'FMGIon\',\'FLG1Ion\',\'FLG2Ion\'])'}),
             ('--cross',       {'type':int,'default':0,'help':'activate cross talk scan 5x5 (default 0)'}),
             ('--asics',      {'type':int,'nargs':'+','choices':range(nAsics),
                               'default':range(nAsics),
                               'help':f'ASICs to use (default {range(nAsics)})'}),
            ]
    scan = ConfigScanBase(userargs=aargs, defargs=defargs)

    args = scan.args

    events = np.abs((args.pulserStart-args.pulserStop)/args.pulserStep) # The 1024 is hardcoded in firmware
    print(f' Number of events: {events}')
    gains_dict={}
    gains_dict['FHGIoff'] =32
    gains_dict['FHGIon']  =36
    gains_dict['FMGIoff'] =40
    gains_dict['FMGIon']  =44
    gains_dict['FLG1Ioff']=1
    gains_dict['FLG1Ion' ]=5
    gains_dict['FLG2Ioff']=33
    gains_dict['FLG2Ion'] =37
    gains_dict['AHG1Ioff'] =16
    gains_dict['AHG1Ion'] =20
    gains_dict['AHG2Ioff'] =48
    gains_dict['AHG2Ion'] =52
    gains_dict['AMG1Ioff'] =24
    gains_dict['AMG1Ion'] =28
    gains_dict['AMG2Ioff'] =56
    gains_dict['AMG2Ion'] =60    
    
    keys = []
    keys.append(f'{args.detname}:user.App.VINJ_DAC.enable')
    keys.append(f'{args.detname}:user.App.VINJ_DAC.dacEn')
    keys.append(f'{args.detname}:user.App.VINJ_DAC.rampEn')
    keys.append(f'{args.detname}:user.App.VINJ_DAC.dacStartValue')
    keys.append(f'{args.detname}:user.App.VINJ_DAC.dacStopValue')
    keys.append(f'{args.detname}:user.App.VINJ_DAC.dacStepValue')
 
    keys.append(f'{args.detname}:user.Gain.SetSameGain4All')
    keys.append(f'{args.detname}:user.Gain.UsePixelMap')
    keys.append(f'{args.detname}:user.Gain.SetGainValue')  
    keys.append(f'{args.detname}:user.Gain.UsePixelMap')  
    keys.append(f'{args.detname}:user.Gain.PixelBitMapSel')  
     
    
    keys.append(f'{args.detname}:expert.App.WaveformControl.InjEn')   
    
    def steps():
        # The metad goes into the step_docstring of the timing DRP's BeginStep data
        # The step_docstring is used to guide the offline calibration routine
        metad = {'detname'    : args.detname,
                 'scantype'   : args.scantype,
                 'events'     : args.events,
        }
        
        d = {}
        d[f'{args.detname}:user.App.VINJ_DAC.enable']         = 1
        d[f'{args.detname}:user.App.VINJ_DAC.dacEn']          = 1
        d[f'{args.detname}:user.App.VINJ_DAC.rampEn']         = 1
        d[f'{args.detname}:user.App.VINJ_DAC.dacStartValue']  = args.pulserStart
        d[f'{args.detname}:user.App.VINJ_DAC.dacStopValue']   = args.pulserStop
        d[f'{args.detname}:user.App.VINJ_DAC.dacStepValue']   = args.pulserStep
        
        d[f'{args.detname}:user.Gain.SetSameGain4All']        = 1
        d[f'{args.detname}:expert.App.WaveformControl.InjEn'] = 1
        
        if "," in args.gain_modes: 
            gain_modes=args.gain_modes.split(',')
        else: 
            gain_modes=[args.gain_modes]
            
        step = 0
        if args.cross:
            pathPll = '/tmp/'
            d[f'{args.detname}:user.Gain.UsePixelMap']            = 1
            d[f'{args.detname}:user.Gain.PixelBitMapSel']         = 7
            d[f'{args.detname}:user.Gain.SetGainValue']           = 36

            for gain_mode in gain_modes:
                gain_mode=gain_mode.strip()
                metad['gain_mode'] = gain_mode
                
                off = f'{gain_mode[:-2]}off'
                print(f'gains: {gain_mode}: {gains_dict[gain_mode]} ')
                print(f'gains: {off}: {gains_dict[off]}')
                
                for i0 in range(5):
                    for j0 in range(5):
                        matrix=np.full((168, 192), gains_dict[off])
                        for i in range(i0, 168, 5):
                            for j in range(j0, 192, 5):
                                matrix[i,j]=gains_dict[gain_mode]
                        fn = pathPll+'onthefly.csv'         
                        csvCfg = np.reshape(matrix, (168, 192))
                        np.savetxt(fn, csvCfg, delimiter=',', newline='\n', comments='', fmt='%d')
                        
                        for asic in args.asics:
                            metad['asic'] = asic
                            metad['step'] = step
                            yield (d, float(step), json.dumps(metad))
                            step += 1
                        
        else:
            d[f'{args.detname}:user.Gain.UsePixelMap']            = 0
            d[f'{args.detname}:user.Gain.PixelBitMapSel']         = 1
            for gain_mode in gain_modes:
                gain_mode=gain_mode.strip()
                metad['gain_mode'] = gain_mode
                d[f'{args.detname}:user.Gain.SetGainValue']           = gains_dict[gain_mode]
                #d[f'{args.detname}:user.gain_mode'] = int(gain_mode_value(gain_mode))
                print(f'gain_mode {gain_mode}:{gains_dict[gain_mode]}')
                for asic in args.asics:
                    metad['asic'] = asic
                    metad['step'] = step
                    yield (d, float(step), json.dumps(metad))
                    step += 1

    scan.run(keys,steps)

if __name__ == '__main__':
    main()
