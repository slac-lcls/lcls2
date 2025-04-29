from psdaq.cas.config_scan_base import ConfigScanBase
from psdaq.configdb.epixm320_config import gain_mode_value
import json

def main():

    # default command line arguments
    defargs = {'--events'  :1000,
               '--hutch'   :'mfx',
               '--detname' :'epixuhr_0',
               '--scantype':'pedestal',
               '--record'  :1}

    aargs = [('--gain-modes',  {'type':str,
                               'default':'FHG, FMG, FLG1, FLG2, AHLG1, AHLG2, AMLG1, AMLG2',
                               'help':'Gain modes to use (default [\'FHG \', \'FMG \', \'FLG1 \', \'FLG2 \', \'AHLG1 \', \'AHLG2 \', \'AMLG1 \', \'AMLG2 \'])'}),
            ]
    scan = ConfigScanBase(userargs=aargs, defargs=defargs)

    args = scan.args

    keys = [f'{args.detname}:user.gain_mode']
    
    gains_dict={}
    gains_dict['FHG']   = 32
    gains_dict['FMG']   = 40
    gains_dict['FLG1']  = 1
    gains_dict['FLG2']  = 33
    gains_dict['AHLG1'] = 16
    gains_dict['AHLG2'] = 48
    gains_dict['AMLG1'] = 24
    gains_dict['AMLG2'] = 56
    
    if "," in args.gain_modes: 
        gain_modes=args.gain_modes.split(',')
    else: 
        gain_modes=[args.gain_modes]

    keys = []
    
    keys.append(f'{args.detname}:user.Gain.SetSameGain4All')
    keys.append(f'{args.detname}:user.Gain.UsePixelMap')
    keys.append(f'{args.detname}:user.Gain.SetGainValue')  
    keys.append(f'{args.detname}:user.Gain.UsePixelMap')  
    keys.append(f'{args.detname}:user.Gain.PixelBitMapSel')  
    
    keys.append(f'{args.detname}:user.App.VINJ_DAC.enable')
    keys.append(f'{args.detname}:user.App.VINJ_DAC.dacEn')
    keys.append(f'{args.detname}:user.App.VINJ_DAC.rampEn')
    keys.append(f'{args.detname}:user.App.VINJ_DAC.dacStartValue')
    keys.append(f'{args.detname}:user.App.VINJ_DAC.dacStopValue')
    keys.append(f'{args.detname}:user.App.VINJ_DAC.dacStepValue')

    def steps():
        # The metad goes into the step_docstring of the timing DRP's BeginStep data
        # The step_docstring is used to guide the offline calibration routine
        metad = {'detname' : args.detname,
                 'scantype': args.scantype,
                 'events'  : args.events}
        d = {}
        step = 0
        d[f'{args.detname}:user.Gain.SetSameGain4All']        = 1
        d[f'{args.detname}:user.Gain.UsePixelMap']            = 0
        d[f'{args.detname}:user.Gain.PixelBitMapSel']         = 0
        d[f'{args.detname}:user.App.VINJ_DAC.enable']         = 0
        d[f'{args.detname}:user.App.VINJ_DAC.dacEn']          = 0
        d[f'{args.detname}:user.App.VINJ_DAC.rampEn']         = 0
        d[f'{args.detname}:user.App.VINJ_DAC.dacStartValue']  = 0
        d[f'{args.detname}:user.App.VINJ_DAC.dacStopValue']   = 0
        d[f'{args.detname}:user.App.VINJ_DAC.dacStepValue']   = 0

        for gain_mode in gain_modes:
            gain_mode=gain_mode.strip()
            
            print(f"gain_mode: {gain_mode}")
            print(f"gain value: {gains_dict[gain_mode]}")
            
            metad['gain_mode'] = gain_mode
            metad['step']      = int(step)
            
            d[f'{args.detname}:user.Gain.SetGainValue'] = gains_dict[gain_mode]
            
            yield (d, float(step), json.dumps(metad))
            step += 1

    scan.run(keys,steps)

if __name__ == '__main__':
    main()