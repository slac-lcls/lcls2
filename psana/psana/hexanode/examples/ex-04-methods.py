#!/usr/bin/env python
#------------------------------

import hexanode

#------------------------------

def test_01():
    print('call pure python')

#------------------------------

def test_02():
    print('call cython hexanode.met1()')
    hexanode.met1()

#------------------------------

def test_03():
    print('call hexanode.fib(90)')
    print(hexanode.fib(90))

#------------------------------

def test_04():
    print('call ctest_resort()')
    hexanode.ctest_resort()

#------------------------------

def test_05():
    import numpy as np

    print('test LMF_IO')
    NUMBER_OF_CHANNELS = 32
    NUMBER_OF_HITS = 16

    o = hexanode.lmf_io(NUMBER_OF_CHANNELS, NUMBER_OF_HITS)
    fname = b"/reg/g/psdm/detector/data_test/lmf/hexanode-example-CO_4.lmf"
    o.open_input_lmf(fname)
    print('Start time: %s' % o.start_time())
    print('Stop time : %s' % o.stop_time())

    #text = np.array((512,), dtype=np.int8)
    #text = 'none'
    for errcode in range(10) :
        text = o.get_error_text(errcode)
        print('Test get_error_text(%d): %s' % (errcode, text))

    print('file_path_name: %s' % o.file_path_name)
    print('get_last_level_info', o.get_last_level_info())
    print('get_double_time_stamp', o.get_double_time_stamp())

    print('data_format_in_user_header', o.data_format_in_user_header)
    print('error_flag', o.error_flag)
    print('version_string', o.version_string)
    print('comment', o.comment)
    print('header_size', o.header_size)
    print('number_of_coordinates', o.number_of_coordinates)
    print('timestamp_format', o.timestamp_format)
    print('common_mode', o.common_mode)
    print('uint64_number_of_events', o.uint64_number_of_events)
    print('daq_id', o.daq_id)
    print('tdc_resolution', o.tdc_resolution)

    print('tdc8hp().user_header_version',   o.tdc8hp.user_header_version)
    print('tdc8hp().trigger_channel_p64',   o.tdc8hp.trigger_channel_p64)
    print('tdc8hp().trigger_dead_time_p68', o.tdc8hp.trigger_dead_time_p68)
    print('tdc8hp().group_range_start_p69', o.tdc8hp.group_range_start_p69)
    print('tdc8hp().group_range_end_p70',   o.tdc8hp.group_range_end_p70)

    for i in range(10) :
        if o.read_next_event() :
             print('Event number: %04d number_of_channels: %d'%\
                    (o.get_event_number(), o.get_number_of_channels()))
             nhits = np.zeros((NUMBER_OF_CHANNELS,), dtype=np.int32)
             o.get_number_of_hits_array(nhits)
             print('   number_of_hits_array', nhits[:8])

             dtdc = np.zeros((NUMBER_OF_CHANNELS, NUMBER_OF_HITS), dtype=np.float64)
             o.get_tdc_data_array(dtdc)
             print('   TDC data:\n', dtdc[0:8,0:5])

#------------------------------

def test_06():
    import numpy as np
    print('test sort_class')
    o = hexanode.py_sort_class()
    status, command, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y=\
    hexanode.py_read_config_file(b"/reg/g/psdm/detector/data_test/lmfsorter.txt", o)
    print('read_config_file status, command, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y=',\
                            status, command, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y)

    status = hexanode.py_read_calibration_tables(b"calibration_table.txt", o)

    print('Cu1', o.cu1) 
    print('Cu2', o.cu2) 
    print('Cv1', o.cv1) 
    print('Cv2', o.cv2) 
    print('Cw1', o.cw1) 
    print('Cw2', o.cw2) 
    print('Cmcp',o.cmcp )

    print('use_sum_correction', o.use_sum_correction)
    print('use_pos_correction', o.use_pos_correction)

    o.set_tdc_resolution_ns(0.025)
    o.set_tdc_array_row_length(16)

    nda_count = np.zeros((6,), dtype=np.int32)
    o.set_count(nda_count)
    #o.tdc_pointer = &tdc_ns[0][0];

    print('common_start_mode                ', o.common_start_mode)
    print('use_hex                          ', o.use_hex)
    print('use_mcp                          ', o.use_mcp)
    print('mcp_radius                       ', o.mcp_radius)
    print('uncorrected_time_sum_half_width_u', o.uncorrected_time_sum_half_width_u)
    print('uncorrected_time_sum_half_width_v', o.uncorrected_time_sum_half_width_v)
    print('uncorrected_time_sum_half_width_w', o.uncorrected_time_sum_half_width_w)
    print('dead_time_anode                  ', o.dead_time_anode)
    print('dead_time_mcp                    ', o.dead_time_mcp)          

    for errcode in range(2) :
        text = o.get_error_text(errcode, 512)
        print('Test get_error_text(%d): %s' % (errcode, text))



    #o.shift_sums(+1, offset_sum_u, offset_sum_v, offset_sum_w)
    #o.shift_layer_w(+1, w_offset)
    #o.shift_position_origin(+1, pos_offset_x, pos_offset_y)

    status = o.do_calibration()
    status = o.feed_calibration_data(True, w_offset)

    o.create_scalefactors_calibrator(True, o.runtime_u,\
                                           o.runtime_v,\
                                           o.runtime_w, 0.78,\
                                           o.fu, o.fv, o.fw)

    #print('map_is_full_enough', hexanode.py_sorter_scalefactors_calibration_map_is_full_enough(o))

    #sfo = hexanode.py_scalefactors_calibration_class(o)
    #print('map_is_full_enough()', sfo.map_is_full_enough())
    #print('best_fv', sfo.best_fv)
    #print('best_fw', sfo.best_fw)
    #print('best_w_offset', sfo.best_w_offset)

#------------------------------

def usage(tname):
    s = '\nUsage: python psana/psana/hexanode/examples/ex-04-methods.py <test-number>'
    if tname in ('0',)    : s+='\n 0 - test ALL'
    if tname in ('0','1') : s+='\n 1 - call pure python'
    if tname in ('0','2') : s+='\n 2 - hexanode.met1()'
    if tname in ('0','3') : s+='\n 3 - hexanode.fib()'
    if tname in ('0','4') : s+='\n 4 - call ctest_resort()'
    if tname in ('0','5') : s+='\n 5 - test LMF_IO'
    if tname in ('0','6') : s+='\n 6 - test sort_class'
    return s

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print('%s' % usage(tname))
    print(50*'_', '\nTest %s' % tname)
    if tname in ('0','1') : test_01()
    if tname in ('0','2') : test_02()
    if tname in ('0','3') : test_03()
    if tname in ('0','4') : test_04()
    if tname in ('0','5') : test_05()
    if tname in ('0','6') : test_06()
    print('%s' % usage(tname))
    sys.exit('End of test %s' % tname)

#------------------------------
