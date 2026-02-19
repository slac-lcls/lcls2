from psdaq.cas.config_scan_base import ConfigScanBase
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
                               'default':'FHG, FMG, FLG1, FLG2, AHLG1, AHLG2, AMLG1, AMLG2',
                               'help':'Gain modes to use (default [\'FHG \', \'FMG \', \'FLG1 \', \'FLG2 \', \'AHLG1 \', \'AHLG2 \', \'AMLG1 \', \'AMLG2 \'])'}),
             ('--cross',       {'type':int,'default':0,'help':'activate cross talk scan 5x5 (default 0)'}),
             ('--asics',      {'type':int,'nargs':'+','choices':range(nAsics),
                               'default':range(nAsics),
                               'help':f'ASICs to use (default {range(nAsics)})'}),
            ]
    scan = ConfigScanBase(userargs=aargs, defargs=defargs)

    args = scan.args
        
    gains_dict={}
    
    gains_dict['FHG']    = {'level':36, 'stop': 6400,  'start': 0, 'step':1, 'off':32, }
    gains_dict['FMG']    = {'level':44, 'stop': 10000, 'start': 0, 'step':1, 'off':40, }
    gains_dict['FLG1' ]  = {'level':5,  'stop': 28000, 'start': 0, 'step':1, 'off':1,  }
    gains_dict['FLG2']   = {'level':37, 'stop': 28000, 'start': 0, 'step':1, 'off':33, }
    gains_dict['AHLG1']  = {'level':20, 'stop': 28000, 'start': 0, 'step':1, 'off':16, }
    gains_dict['AHLG2']  = {'level':52, 'stop': 28000, 'start': 0, 'step':1, 'off':48, }
    gains_dict['AMLG1']  = {'level':28, 'stop': 28000, 'start': 0, 'step':1, 'off':24, }
    gains_dict['AMLG2']  = {'level':60, 'stop': 28000, 'start': 0, 'step':1, 'off':56, }
    
    
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
        
            d[f'{args.detname}:user.Gain.UsePixelMap']            = 1
            d[f'{args.detname}:user.Gain.PixelBitMapSel']         = 7
            #d[f'{args.detname}:user.Gain.SetGainValue']           = 36

            for gain_mode in gain_modes:

                gain_mode=gain_mode.strip()
                metad['gain_mode'] = gain_mode
                
        #         off = f'{gain_mode[:-2]}off'
                #print(f'gains: {gain_mode}: {gains_dict[gain_mode]} ')
        #         print(f'gains: {off}: {gains_dict[off]}')
                

                matrix=np.full((168, 192), gains_dict[gain_mode]['off'], dtype=np.uint16)
                
                for i in range(0, 168, 5):
                    for j in range(0, 192, 5):
                        matrix[i,j]=np.uint16(gains_dict[gain_mode]['level'])
                #print(f"MATRIX CREATED, {matrix}")
                mat=', '.join(map(str, matrix.flatten()))
#                for i in matrix:
#                    mat=mat+str(matrix[i])+', '
                #print(mat)

 #               exit
#               # print(mat)
#                result = ", ".join([" ".join(row) for row in mat_list])
 #               print(result)
                d[f'{args.detname}:expert.pixelBitMaps._7_on_the_fly']=mat
                
                events = (gains_dict[gain_mode]['stop'])/gains_dict[gain_mode]['step']
                pulserStart=gains_dict[gain_mode]['start']
                pulserStop=gains_dict[gain_mode]['stop']
                pulserStep=gains_dict[gain_mode]['step']
                print(f'Start:{pulserStart}, Stop:{pulserStop}, Step:{pulserStep}')
                d[f'{args.detname}:user.App.VINJ_DAC.dacStartValue']  = pulserStart
                d[f'{args.detname}:user.App.VINJ_DAC.dacStopValue']   = pulserStop
                d[f'{args.detname}:user.App.VINJ_DAC.dacStepValue']   = pulserStep
                
                #metad['gain_mode'] = gain_mode
                d[f'{args.detname}:user.Gain.SetGainValue']           = gains_dict[gain_mode]['level']
               # print(gains_dict[gain_mode]['level'])
                metad['step'] = step
                metad['events']=events
                yield (d, float(step), json.dumps(metad))
                step += 1
        #                 fn = pathPll+'onthefly.csv'         
        #                 csvCfg = np.reshape(matrix, (168, 192))
        #                 np.savetxt(fn, csvCfg, delimiter=',', newline='\n', comments='', fmt='%d')
                        
        #                 for asic in args.asics:
        #                     metad['asic'] = asic
        #                     metad['step'] = step
        #                     yield (d, float(step), json.dumps(metad))
        #                     step += 1
                        
        else:
            d[f'{args.detname}:user.Gain.UsePixelMap']            = 0
            d[f'{args.detname}:user.Gain.PixelBitMapSel']         = 1
            for gain_mode in gain_modes:
                gain_mode=gain_mode.strip()
                print(f"gain mode: {gain_mode}")
                events = (gains_dict[gain_mode]['stop'])/gains_dict[gain_mode]['step']
                pulserStart=gains_dict[gain_mode]['start']
                pulserStop=gains_dict[gain_mode]['stop']
                pulserStep=gains_dict[gain_mode]['step']
                print(f'Start:{pulserStart}, Stop:{pulserStop}, Step:{pulserStep}')
                d[f'{args.detname}:user.App.VINJ_DAC.dacStartValue']  = pulserStart
                d[f'{args.detname}:user.App.VINJ_DAC.dacStopValue']   = pulserStop
                d[f'{args.detname}:user.App.VINJ_DAC.dacStepValue']   = pulserStep
                
                metad['gain_mode'] = gain_mode
                d[f'{args.detname}:user.Gain.SetGainValue']           = gains_dict[gain_mode]['level']
                print(gains_dict[gain_mode]['level'])
                metad['step'] = step
                metad['events']=events
                yield (d, float(step), json.dumps(metad))
                step += 1

    scan.run(keys,steps)

if __name__ == '__main__':
    main()
