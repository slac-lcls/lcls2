# manually set the andor exposure time to 2 virtual timeslots (2ms)? should
# be large enough to capture the laser (including the unknown UED
# "setup time" between the eventcode arrival and beam/laser arrival)

# DAQ:UED:XPM:0:PART:0:SeqMask must be set to 3 so xpm starts both
# sequencing engines at the same time

# kukhee and patrick kramer will use this for laser timing:
# TPG Sequence Engine 7, Bit 5 for 10 Hz starting from TS01 (eventcode 0x75)

# need the following 5 event codes:
# sequence start: use to trigger laser/ebeam delay-generators for shutters
# andor start exposure
# andor timestamp (general daq trigger)
# laser marker
# ebeam marker

#pre_delay=1; L_dark=2; M_pre=5; N_post=5; engine = 0

# first engine
daq_trigger_ec = 0
andor_expose_ec = 1

# second engine
start_seq_ec = 0
laser_marker_ec = 1
ebeam_pre_ec = 2
ebeam_post_ec = 3

# for dark, pre-ebeam-shots, laser, post-ebeam-shots
if engine==0:
    eventcodes = {'dark':[daq_trigger_ec],'pre':[daq_trigger_ec],
                  'laser':[daq_trigger_ec],'post':[daq_trigger_ec]}
else:
    # put in placeholder_ec since I'm not sure if Matt supports
    # calling ControlRequest with an empty list
    eventcodes = {'dark':[], 'pre':[ebeam_pre_ec],
                  'laser':[laser_marker_ec],'post':[ebeam_post_ec]}

if engine==0:
    seqcodes = {daq_trigger_ec: 'DAQ Trigger', andor_expose_ec: 'Andor Expose'}
else:
    seqcodes = {start_seq_ec: 'Start Sequence', ebeam_pre_ec: 'Ebeam Pre Marker', ebeam_post_ec: 'Ebeam Post Marker', laser_marker_ec: 'Laser Marker'}

instrset = []

#def trigger(codes):
#    # sync with 10Hz laser
#    instrset.append( ACRateSync( 1, "10H", occ=1 ) )
#    # start andor exposure if we're engine 0
#    if engine==0: instrset.append( ControlRequest([andor_expose_ec]) )
#    # wait one virtual 1080Hz timeslot to match laser TS01
#    instrset.append( FixedRateSync( "500kH", occ=463 ) )
#    # daq trigger must come before andor image-transmission completion
#    # this is an IOC-timestamping ordering requirement
#    instrset.append( ControlRequest(codes) )

instrset.append( ACRateSync( 1, "10H", occ=1 ) ) # sync with 10Hz laser
if engine==1:
    instrset.append( ControlRequest([start_seq_ec]) ) # early TTL pulse for DGs

# "predelay": change occ to allow time for slow shutters to open
instrset.append( ACRateSync( 1, "10H", occ=pre_delay ) )

# L_dark dark shots
for i in range(L_dark):
    #trigger(eventcodes['dark']) # no shutter markers
    # sync with 10Hz laser
    instrset.append( ACRateSync( 1, "10H", occ=1 ) )
    # start andor exposure if we're engine 0
    if engine==0: instrset.append( ControlRequest([andor_expose_ec]) )
    # wait one virtual 1080Hz timeslot to match laser TS01
    instrset.append( FixedRateSync( "500kH", occ=463 ) )
    # daq trigger must come before andor image-transmission completion
    # this is an IOC-timestamping ordering requirement
    if len(eventcodes['dark'])>0: instrset.append( ControlRequest(eventcodes['dark']) )

# M_pre "pre-laser" ebeam shots
for i in range(M_pre):
    #trigger(eventcodes['pre'])
    # sync with 10Hz laser
    instrset.append( ACRateSync( 1, "10H", occ=1 ) )
    # start andor exposure if we're engine 0
    if engine==0: instrset.append( ControlRequest([andor_expose_ec]) )
    # wait one virtual 1080Hz timeslot to match laser TS01
    instrset.append( FixedRateSync( "500kH", occ=463 ) )
    # daq trigger must come before andor image-transmission completion
    # this is an IOC-timestamping ordering requirement
    instrset.append( ControlRequest(eventcodes['pre']) )

# 1 laser/ebeam shots
#trigger(eventcodes['laser'])
# sync with 10Hz laser
instrset.append( ACRateSync( 1, "10H", occ=1 ) )
# start andor exposure if we're engine 0
if engine==0: instrset.append( ControlRequest([andor_expose_ec]) )
# wait one virtual 1080Hz timeslot to match laser TS01
instrset.append( FixedRateSync( "500kH", occ=463 ) )
# daq trigger must come before andor image-transmission completion
# this is an IOC-timestamping ordering requirement
instrset.append( ControlRequest(eventcodes['laser']) )

# N_post "pre-laser" ebeam shots
for i in range(N_post):
    #trigger(eventcodes['post'])
    # sync with 10Hz laser
    instrset.append( ACRateSync( 1, "10H", occ=1 ) )
    # start andor exposure if we're engine 0
    if engine==0: instrset.append( ControlRequest([andor_expose_ec]) )
    # wait one virtual 1080Hz timeslot to match laser TS01
    instrset.append( FixedRateSync( "500kH", occ=463 ) )
    # daq trigger must come before andor image-transmission completion
    # this is an IOC-timestamping ordering requirement
    instrset.append( ControlRequest(eventcodes['post']) )

instrset.append( Branch.unconditional(0) )
