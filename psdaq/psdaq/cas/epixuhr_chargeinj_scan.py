from psdaq.cas.config_scan_base import ConfigScanBase
from psdaq.configdb.epixm320_config import gain_mode_value
import numpy as np
import json
import logging

nAsics = 4
nColumns = 384

def main():

    # default command line arguments
    defargs = {'--hutch'   : 'asc',
               '--detname' : 'epixuhr_0',
               '--scantype': 'chargeinj',
               '--record'  : 1,
               }

    aargs = [('--gain-modes',  {'type':str,
                               'default':'FHGIon, FMGIon, FLG1Ion, FLG2Ion, AHLG1Ion, AHLG2Ion, AMLG1Ion, AMLG2Ion',
                               'help':'Gain modes to use (default [\'FHGIon\', \'FMGIon\', \'FLG1Ion\', \'FLG2Ion\', \'AHLG1Ion\', \'AHLG2Ion\', \'AMLG1Ion\', \'AMLG2Ion\'])'}),
             ('--cross',       {'type':int,'default':0,'help':'activate cross talk scan 5x5 (default 0)'}),
             ('--asics',      {'type':int,'nargs':'+','choices':range(nAsics),
                               'default':range(nAsics),
                               'help':f'ASICs to use (default {range(nAsics)})'}),
            ]
    scan = ConfigScanBase(userargs=aargs, defargs=defargs)

    args = scan.args
        
    gains_dict={}
    gains_dict['FHGIoff']   = {'level':32, 'stop': 6400,  'start': 0, 'step':1}
    gains_dict['FHGIon']    = {'level':36, 'stop': 6400,  'start': 0, 'step':1}
    gains_dict['FMGIoff']   = {'level':40, 'stop': 10000, 'start': 0, 'step':1}
    gains_dict['FMGIon']    = {'level':44, 'stop': 10000, 'start': 0, 'step':1}
    gains_dict['FLG1Ioff']  = {'level':1,  'stop': 28000, 'start': 0, 'step':1}
    gains_dict['FLG1Ion' ]  = {'level':5,  'stop': 28000, 'start': 0, 'step':1}
    gains_dict['FLG2Ioff']  = {'level':33, 'stop': 28000, 'start': 0, 'step':1}
    gains_dict['FLG2Ion']   = {'level':37, 'stop': 28000, 'start': 0, 'step':1}
    gains_dict['AHLG1Ioff'] = {'level':16, 'stop': 28000, 'start': 0, 'step':1}
    gains_dict['AHLG1Ion']  = {'level':20, 'stop': 28000, 'start': 0, 'step':1}
    gains_dict['AHLG2Ioff'] = {'level':48, 'stop': 28000, 'start': 0, 'step':1}
    gains_dict['AHLG2Ion']  = {'level':52, 'stop': 28000, 'start': 0, 'step':1}
    gains_dict['AMLG1Ioff'] = {'level':24, 'stop': 28000, 'start': 0, 'step':1}
    gains_dict['AMLG1Ion']  = {'level':28, 'stop': 28000, 'start': 0, 'step':1}
    gains_dict['AMLG2Ioff'] = {'level':56, 'stop': 28000, 'start': 0, 'step':1}
    gains_dict['AMLG2Ion']  = {'level':60 ,'stop': 28000, 'start': 0, 'step':1}
    
    
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
                print(gain_mode)
                args.events = (gains_dict[gain_mode]['stop'])/gains_dict[gain_mode]['step']
                args.pulserStart=gains_dict[gain_mode]['start']
                args.pulserStop=gains_dict[gain_mode]['stop']
                args.pulserStep=gains_dict[gain_mode]['step']
                print(f'{args.pulserStart}, {args.pulserStop}, {args.pulserStep}')
                
                d[f'{args.detname}:user.App.VINJ_DAC.dacStartValue']  = args.pulserStart
                d[f'{args.detname}:user.App.VINJ_DAC.dacStopValue']   = args.pulserStop
                d[f'{args.detname}:user.App.VINJ_DAC.dacStepValue']   = args.pulserStep
                
                metad['gain_mode'] = gain_mode
                d[f'{args.detname}:user.Gain.SetGainValue']           = gains_dict[gain_mode]['level']
                print(gains_dict[gain_mode]['level'])
                metad['step'] = step
                yield (d, float(step), json.dumps(metad))
                step += 1

    scan.run(keys,steps)

if __name__ == '__main__':
    main()
