from psdaq.seq.periodicgenerator import *
from psdaq.seq.seqwrite import *

def ctrl_write(name, instr, output):
    seq_write_py  (instr, output)
    seq_write_json(name, instr, output)

def main():

    parser = argparse.ArgumentParser(description='train pattern generator')
    parser.add_argument("-o", "--output", required=True , help="file output path")
    args = parser.parse_args()
    ppath = args.output

    try:
        os.mkdir(ppath)
    except:
        pass

    #  Count in steps of 35.7 kHz
    period_360H = int((35000/0.98)//360 + 1)
    period_120H = int(3*period_360H)
    period_60H = int(2*period_120H)
    period_30H = int(2*period_60H)
    period_10H = int(3*period_30H)
    period_5H  = int(2*period_10H)
    period_1H  = int(5*period_5H)
    period_0_5H = int(2*period_1H)

    c = {}
    # 8-11
    TS=0
    c[2] = PeriodicGenerator([period_120H,period_120H,period_120H,period_60H],
                             [TS]*4,
                             marker='35kH')
    # 12-15
    c[3] = PeriodicGenerator([period_30H,period_10H,period_5H,period_1H],
                             [TS]*4,
                             marker='35kH')
    # 16-19
    c[4] = PeriodicGenerator([period_0_5H],
                             [TS],
                             marker='35kH')
    # 20-23
    TS=period_360H
    c[5] = PeriodicGenerator([period_120H,period_60H,period_30H,period_10H],
                             [TS]*4,
                             marker='35kH')
    # 24-27
    c[6] = PeriodicGenerator([period_5H,period_1H,period_0_5H],
                             [TS]*3,
                             marker='35kH')
    # 28-31
    TS=int(2*period_360H)
    c[7] = PeriodicGenerator([period_120H,period_120H,period_120H,period_60H],
                             [TS]*4,
                             marker='35kH')
    # 32-35
    c[8] = PeriodicGenerator([period_30H,period_10H,period_5H,period_1H],
                             [TS]*4,
                             marker='35kH')
    # 36-39
    c[9] = PeriodicGenerator([period_0_5H],
                             [TS],
                             marker='35kH')
    # 40-43
    TS=int(3*period_360H)
    c[10] = PeriodicGenerator([period_120H,period_60H,period_30H,period_10H],
                              [TS]*4,
                              marker='35kH')
    # 44-47
    c[11] = PeriodicGenerator([period_5H,period_1H,period_0_5H],
                              [TS]*3,
                              marker='35kH')
    # 48-51
    TS=int(4*period_360H)
    c[12] = PeriodicGenerator([period_120H,period_120H,period_120H,period_60H],
                              [TS]*4,
                              marker='35kH')
    # 52-55
    c[13] = PeriodicGenerator([period_30H,period_10H,period_5H,period_1H],
                              [TS]*4,
                              marker='35kH')
    # 56-59
    c[14] = PeriodicGenerator([period_0_5H],
                              [TS],
                              marker='35kH')
    # 60-63
    TS=int(5*period_360H)
    c[15] = PeriodicGenerator([period_120H,period_60H,period_30H,period_10H],
                              [TS]*4,
                              marker='35kH')
    # 64-67
    c[16] = PeriodicGenerator([period_5H,period_1H,period_0_5H],
                              [TS]*3,
                              marker='35kH')

    # goose trigger?
    c[60] = PeriodicGenerator([35500,35500],[0,period_120H],marker='35kH')

    for key,gen in c.items():
        ctrl_write(name=f'c{key}',
                   instr=gen.instr,
                   output=f'{ppath}/c{key}')

if __name__=='__main__':
    main()

