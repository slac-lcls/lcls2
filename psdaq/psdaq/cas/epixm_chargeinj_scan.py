from psdaq.cas.config_scan_base import ConfigScanBase
import numpy as np
import json

nAsics = 4
nColumns = 384

def main():

    aargs = [('--spacing',{'type':int,'default':5,'help':'size of lane'})]
    scan = ConfigScanBase(aargs)

    args = scan.args
    keys = []
    keys.append(f'{args.detname}:user.gain_mode')
    keys.append(f'{args.detname}:user.chgInj_column_map')
    for a in range(4):
        saci = f'{args.detname}:expert.App.Mv2Asic[{a}]'
        keys.append(f'{saci}.CompTH_ePixM')
        keys.append(f'{saci}.Precharge_DAC_ePixM')
        keys.append(f'{saci}.test')
        keys.append(f'{saci}.Pulser')

    # scan loop
    spacing  = args.spacing

    def gain_mode_map(gain_mode):
        compTH        = (  0,   63,    12,      0 )[gain_mode] # SoftHigh/SoftLow/Auto/User
        precharge_DAC = ( 45,   50,    45,      0 )[gain_mode]
        name          = ('SH', 'SL', 'AHL', 'User')[gain_mode]
        return (compTH, precharge_DAC, name)

    def column_map(spacing, position):
        firstCol = position
        lastCol = min(position + spacing, nColumns-1)
        lane_selected = np.zeros(nColumns, dtype=np.uint8)
        lane_selected[firstCol : lastCol + 1] = 1
        return lane_selected

    def steps():
        d = {}
        metad = {}
        metad['detname'] = args.detname
        metad['scantype'] = 'chargeinj'
        d[f'{args.detname}:user.gain_mode'] = 3  # User
        for a in range(nAsics):
            saci = f'{args.detname}:expert.App.Mv2Asic[{a}]'
            d[f'{saci}.test'] = 1
            d[f'{saci}.Pulser'] = 0xc8
            # d[f'{saci}:PulserSync'] = 1  # with ghost correction
        step = 0
        for gain_mode in range(3):
            compTH, precharge_DAC, name = gain_mode_map(gain_mode)
            metad['gain_mode'] = name
            for a in range(nAsics):
                saci = f'{args.detname}:expert.App.Mv2Asic[{a}]'
                d[f'{saci}.CompTH_ePixM'] = compTH
                d[f'{saci}.Precharge_DAC_ePixM'] = precharge_DAC
            for column in range(0, nColumns, spacing):
                cmap = column_map(spacing, column)
                #  Do I need to convert to list and lose the dimensionality? (json not serializable)
                d[f'{args.detname}:user.chgInj_column_map'] = cmap.tolist()
                #d[f'{args.detname}:user.chgInj_column_map'] = cmap
                #  Set the global meta data
                metad['step'] = step
                step += 1

                yield (d, float(step), json.dumps(metad))

    scan.run(keys,steps)

if __name__ == '__main__':
    main()
