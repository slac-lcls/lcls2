from psdaq.cas.config_scan_base import ConfigScanBase
from psdaq.configdb.epixm320_config import gain_mode_value
import json

def main():

    # default command line arguments
    defargs = {'--events'  :1000,
               '--hutch'   :'rix',
               '--detname' :'epixm_0',
               '--scantype':'pedestal',
               '--record'  :1,
               '--run_type':'DARK'}

    aargs = [('--gain-modes', {'type':str,'nargs':'+','choices':['SH','SL','AHL','User'],
                               'default':['AHL','SH'],
                               'help':'Gain modes to use (default [\'AHL\',\'SH\'])'}),]
    scan = ConfigScanBase(userargs=aargs, defargs=defargs)

    args = scan.args

    keys = [f'{args.detname}:user.gain_mode']

    def steps():
        # The metad goes into the step_docstring of the timing DRP's BeginStep data
        # The step_docstring is used to guide the offline calibration routine
        metad = {'detname' : args.detname,
                 'scantype': args.scantype,
                 'events'  : args.events}
        d = {}
        step = 0
        for gain_mode in args.gain_modes:
            gain = gain_mode_value(gain_mode)
            metad['gain_mode'] = gain_mode
            metad['step']      = int(gain)
            d[f'{args.detname}:user.gain_mode'] = int(gain)
            yield (d, float(step), json.dumps(metad))
            step += 1

    scan.run(keys,steps)

if __name__ == '__main__':
    main()
