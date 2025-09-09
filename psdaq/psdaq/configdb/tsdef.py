import socket

#  One authoritative source for the ordering of the timing system markers

if 'ued' in socket.gethostname():
    fixedRates  = ['500kHz', '100kHz','50kHz','10kHz','5kHz','1kHz','500Hz','1Hz']
    fixedRateHzToMarker = {'500kHz':0, '100kHz':1, '50kHz':2, '10kHz':3, '5kHz':4, '1kHz':5, '500Hz':6, '1Hz':7}
    FixedIntvs = [1, 5, 10, 50, 100, 500, 1000, 500000]
    FixedIntvsDict = {"500kH":{"intv":1     ,"marker":0}, 
                      "100kH":{"intv":5     ,"marker":1}, 
                      "50kH" :{"intv":10    ,"marker":2}, 
                      "10kH" :{"intv":50    ,"marker":3}, 
                      "5kH"  :{"intv":100   ,"marker":4}, 
                      "1kH"  :{"intv":500   ,"marker":5}, 
                      "500H" :{"intv":1000  ,"marker":6}, 
                      "1H"   :{"intv":500000,"marker":7}}

    acRates     = ['60Hz','30Hz','10Hz','5Hz','1Hz','0.5Hz']
    acTS        = ['TS%u'%(i+1) for i in range(6)]
    acRateHzToMarker    = {'60Hz':0, '30Hz':1, '10Hz':2, '5Hz':3, '1Hz':4, '0_5Hz':5 }
    ACIntvs    = [1, 2, 6, 12, 60, 120]
    
    ACIntvsDict = {"0.5H":{"intv":120,"marker":5}, 
                   "1H"  :{"intv":60 ,"marker":4}, 
                   "5H"  :{"intv":12 ,"marker":3}, 
                   "10H" :{"intv":6  ,"marker":2}, 
                   "30H" :{"intv":2  ,"marker":1}, 
                   "60H" :{"intv":1  ,"marker":0}}

    FixedFidRate  = 500e3            # seqplot
    FixedToACFids = int(500e3/360)   # needed for seqplot simulation

else:
    fixedRates  = ['1.02Hz','10.2Hz','102Hz','1.02kHz','10.2kHz','71.4kHz','929kHz', 'Undef7', 'Undef8', 'Undef9' ]
    fixedRateHzToMarker = {'929kHz':6, '71kHz':5, '10kHz':4, '1kHz':3, '100Hz':2, '10Hz':1, '1Hz':0}
    FixedIntvs = [910000, 91000, 9100, 910, 91, 13, 1]
    FixedIntvsDict = {"1H"   :{"intv":910000,"marker":0}, 
                      "10H"  :{"intv":91000 ,"marker":1}, 
                      "100H" :{"intv":9100  ,"marker":2}, 
                      "1kH"  :{"intv":910   ,"marker":3}, 
                      "10kH" :{"intv":91    ,"marker":4}, 
                      "70kH" :{"intv":13    ,"marker":5}, 
                      "910kH":{"intv":1     ,"marker":6}}

    acRates     = ['0.5Hz','1Hz','5Hz','10Hz','30Hz','60Hz']
    acTS        = ['TS%u'%(i+1) for i in range(6)]
    acRateHzToMarker    = {'60Hz':5, '30Hz':4, '10Hz':3, '5Hz':2, '1Hz':1, '0_5Hz':0 }
    ACIntvs    = [120, 60, 12, 6, 2, 1]
    
    ACIntvsDict = {"0.5H":{"intv":120,"marker":0}, 
                   "1H"  :{"intv":60 ,"marker":1}, 
                   "5H"  :{"intv":12 ,"marker":2}, 
                   "10H" :{"intv":6  ,"marker":3}, 
                   "30H" :{"intv":2  ,"marker":4}, 
                   "60H" :{"intv":1  ,"marker":5}}

    FixedFidRate  = 910e3                 # seqplot
    FixedToACFids = int(910e3/0.98/360)   # needed for seqplot simulation

