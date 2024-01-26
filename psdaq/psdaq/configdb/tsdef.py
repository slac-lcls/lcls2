#  One authoritative source for the ordering of the timing system markers
fixedRates  = ['1.02Hz','10.2Hz','102Hz','1.02kHz','10.2kHz','71.4kHz','929kHz']
acRates     = ['0.5Hz','1Hz','5Hz','10Hz','30Hz','60Hz']
acTS        = ['TS%u'%(i+1) for i in range(6)]
fixedRateHzToMarker = {'929kHz':6, '71kHz':5, '10kHz':4, '1kHz':3, '100Hz':2, '10Hz':1, '1Hz':0}
acRateHzToMarker    = {'60Hz':5, '30Hz':4, '10Hz':3, '5Hz':2, '1Hz':1, '0.5Hz':0 }
FixedIntvs = [910000, 91000, 9100, 910, 91, 13, 1]
ACIntvs    = [120, 60, 12, 6, 2, 1]

FixedIntvsDict = {"1H"   :{"intv":910000,"marker":0}, 
                  "10H"  :{"intv":91000 ,"marker":1}, 
                  "100H" :{"intv":9100  ,"marker":2}, 
                  "1kH"  :{"intv":910   ,"marker":3}, 
                  "10kH" :{"intv":91    ,"marker":4}, 
                  "70kH" :{"intv":13    ,"marker":5}, 
                  "910kH":{"intv":1     ,"marker":6}}

ACIntvsDict = {"0.5H":{"intv":910000,"marker":0}, 
               "1H"  :{"intv":91000 ,"marker":1}, 
               "5H"  :{"intv":9100  ,"marker":2}, 
               "10H" :{"intv":910   ,"marker":3}, 
               "30H" :{"intv":91    ,"marker":4}, 
               "60H" :{"intv":13    ,"marker":5}}


