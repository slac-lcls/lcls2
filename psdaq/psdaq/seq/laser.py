from psdaq.seq.seq import *

title = 'OPCPA Laser Test'

descset = ['93kHz','33kHz','9.3kHz',]

instrset = []

instrset.append( ControlRequest(3) )

# loop: req 1 of step 10 and repeat 1

start = len(instrset)

instrset.append( FixedRateSync(marker="910kH", occ=10 ) )

instrset.append( ControlRequest(1) )

instrset.append( Branch.conditional(start, 0, 1) )

instrset.append( FixedRateSync(marker="910kH", occ=8 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=2 ) )

instrset.append( ControlRequest(1) )

# loop: req 1 of step 10 and repeat 1

start = len(instrset)

instrset.append( FixedRateSync(marker="910kH", occ=10 ) )

instrset.append( ControlRequest(1) )

instrset.append( Branch.conditional(start, 0, 1) )

instrset.append( FixedRateSync(marker="910kH", occ=6 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=4 ) )

instrset.append( ControlRequest(1) )

# loop: req 1 of step 10 and repeat 1

start = len(instrset)

instrset.append( FixedRateSync(marker="910kH", occ=10 ) )

instrset.append( ControlRequest(1) )

instrset.append( Branch.conditional(start, 0, 1) )

instrset.append( FixedRateSync(marker="910kH", occ=4 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=6 ) )

instrset.append( ControlRequest(1) )

# loop: req 1 of step 10 and repeat 1

start = len(instrset)

instrset.append( FixedRateSync(marker="910kH", occ=10 ) )

instrset.append( ControlRequest(1) )

instrset.append( Branch.conditional(start, 0, 1) )

instrset.append( FixedRateSync(marker="910kH", occ=2 ) )

instrset.append( ControlRequest(2) )

instrset.append( FixedRateSync(marker="910kH", occ=8 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=10 ) )

instrset.append( ControlRequest(1) )

instrset.append( FixedRateSync(marker="910kH", occ=10 ) )

instrset.append( Branch.unconditional(0) )
