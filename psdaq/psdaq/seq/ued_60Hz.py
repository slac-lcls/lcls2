# UED 60Hz Hz setup
seqcodes = {0: '60 Hz VTS1'}

instrset = []
instrset.append( ACRateSync( 0x1, "60H", occ=1) )
instrset.append( FixedRateSync( marker="500kH", occ=463 ) )
instrset.append( ControlRequest([0]) )
instrset.append( FixedRateSync( marker="500kH", occ=463 ) )
instrset.append( Branch.unconditional(0) )
