# Utility functions for the ePixM

def gain_mode_map(gain_mode):
    if gain_mode > 2:
        print(f"gain_mode_map: Bad gain mode {gain_mode}")
        return (0, 0, 'User')

    compTH        = ( 0,   44,   24)   [gain_mode] # SoftHigh/SoftLow/Auto
    precharge_DAC = (45,   45,   45)   [gain_mode]
    name          = ('SH', 'SL', 'AHL')[gain_mode]
    return (compTH, precharge_DAC, name)

