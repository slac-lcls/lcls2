from psdaq.seq.seq import *

#args Namespace(period=[28, 26], start_bucket=[0, 0])
#period 364  args.period [28, 26]
#53 instructions
instrset = []

instrset.append( ControlRequest(3) )

instrset.append( FixedRateSync(marker="910kH", occ=26 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=2 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=24 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=4 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=22 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=6 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=20 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=8 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=18 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=10 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=16 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=12 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=14 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=14 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=12 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=16 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=10 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=18 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=8 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=20 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=6 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=22 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=4 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=24 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=2 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=26 ) )

instrset.append( Branch.unconditional(0) )

seqcodes = {0:'33kHz base',1:'35kHz base'}
