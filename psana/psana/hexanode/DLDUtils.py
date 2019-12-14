#!/usr/bin/env python
"""
Module :py:class:`DLDUtils` 
=========================================================================

    from psana.hexanode.DLDUtils import ...


Created on 2019-12-12 by Mikhail Dubrovin
"""
#----------

USAGE = 'Run example: python .../psana/hexanode/examples/ex-16-proc-data.py'

#----------

import logging
logger = logging.getLogger(__name__)

import sys
#from time import time
#import numpy as np

#----------

def load_int(line) :
    return int(line.strip().split()[0]) # .atoi()


def load_double(line) :
    return float(line.strip().split()[0]) # .atof()


def load_double_couple(line) :
    flds = line.strip().split()
    return float(flds[0]), float(flds[1])


def load_config_pars(txt_config, sorter=None) :
    """in stead of py_read_config_file"""

    if not txt_config : return False

    lines = txt_config.split('\n')
    lins = [l for l in lines if l] # discard empty lines

    #for line in lins : print('==:', line)

    command = load_int(lins[0])
    if command == -1 : return False

    use_hex           = load_int(lins[1]) > 0
    common_start_mode = load_int(lins[2]) == 0
    cu1               = load_int(lins[3]) - 1
    cu2               = load_int(lins[4]) - 1
    cv1               = load_int(lins[5]) - 1
    cv2               = load_int(lins[6]) - 1
    cw1               = load_int(lins[7]) - 1
    cw2               = load_int(lins[8]) - 1
    cmcp              = load_int(lins[9]) - 1
    use_mcp           = cmcp > -1
    offset_sum_u      = load_double(lins[10])
    offset_sum_v      = load_double(lins[11])
    offset_sum_w      = load_double(lins[12])

    pos_offset_x      = load_double(lins[13])
    pos_offset_y      = load_double(lins[14])

    uncorrected_time_sum_half_width_u = load_double(lins[15])
    uncorrected_time_sum_half_width_v = load_double(lins[16])
    uncorrected_time_sum_half_width_w = load_double(lins[17])

    fu = 0.5*load_double(lins[18])
    fv = 0.5*load_double(lins[19])
    fw = 0.5*load_double(lins[20])

    w_offset   = load_double(lins[21]) 
    runtime_u  = load_double(lins[22]) 
    runtime_v  = load_double(lins[23]) 
    runtime_w  = load_double(lins[24]) 

    mcp_radius = load_double(lins[25]) 

    dead_time_anode = load_double(lins[26])
    dead_time_mcp   = load_double(lins[27])
    use_sum_correction = load_int(lins[28]) != 0
    use_pos_correction = load_int(lins[29]) != 0

    check_input = load_int(lins[30])

    print('offset_sum_u: %6.1f' % offset_sum_u)
    print('offset_sum_v: %6.1f' % offset_sum_v)
    print('offset_sum_w: %6.1f' % offset_sum_w)

    print('uncorrected_time_sum_half_width_u: %4.1f' % uncorrected_time_sum_half_width_u)
    print('uncorrected_time_sum_half_width_v: %4.1f' % uncorrected_time_sum_half_width_v)
    print('uncorrected_time_sum_half_width_w: %4.1f' % uncorrected_time_sum_half_width_w)

    print('runtime_u: %6.1f' % runtime_u)
    print('runtime_v: %6.1f' % runtime_v)
    print('runtime_w: %6.1f' % runtime_w)

    print('use_sum_correction: %s' % use_sum_correction)
    print('use_pos_correction: %s' % use_pos_correction)

    if check_input != 88888 : 
        logger.warning("Configuration file was not correctly read.")
        # close file
        # delete sorter
        return sys.exit("Configuration file was not correctly read.")

    if sorter is not None:
        # pass values to sorter sorter
        sorter.Cu1 = cu1
        sorter.Cu2 = cu2
        sorter.Cv1 = cv1
        sorter.Cv2 = cv2
        sorter.Cw1 = cw1
        sorter.Cw2 = cw2
        sorter.Cmcp= cmcp
        sorter.use_MCP = use_mcp

        sorter.fu = fu
        sorter.fv = fv
        sorter.fw = fw
        sorter.w_offset = w_offset

        sorter.runtime_u = runtime_u
        sorter.runtime_v = runtime_v
        sorter.runtime_w = runtime_w
        sorter.MCP_radius = mcp_radius

        sorter.dead_time_anode = dead_time_anode
        sorter.dead_time_mcp = dead_time_mcp
        sorter.use_sum_correction = use_sum_correction
        sorter.use_pos_correction = use_pos_correction

    return command, offset_sum_u, offset_sum_v, offset_sum_w,\
           w_offset, pos_offset_x, pos_offset_y

#----------

def load_calibration_group(lins, il, cmt='') :
    points = load_int(lins[il])
    print('points: %d for %s' % (points, cmt))

    list_of_pairs = []
    for i in range(points) :
        il += 1
        x,y = load_double_couple(lins[il])
        print('l:%2d x=%8.3f y=%8.3f' % (i,x,y))
    il += 1
    return list_of_pairs, il


def load_calibration_tables(txt_calib, sorter=None) :
    """in stead of py_read_calibration_tables"""

    if not txt_calib : return False

    lines = txt_calib.split('\n')
    lins = [l for l in lines if l] # discard empty lines
    #for line in lins : print('==:', line)

    il = 0
    sum_corrector_U, il = load_calibration_group(lins, il, cmt='sum_corrector_U')
    sum_corrector_V, il = load_calibration_group(lins, il, cmt='sum_corrector_V')
    sum_corrector_W, il = load_calibration_group(lins, il, cmt='sum_corrector_W')\
      if sorter is not None and sorter.use_HEX else ([],il)

    pos_corrector_U, il = load_calibration_group(lins, il, cmt='pos_corrector_U')
    pos_corrector_V, il = load_calibration_group(lins, il, cmt='pos_corrector_V')
    pos_corrector_W, il = load_calibration_group(lins, il, cmt='pos_corrector_W')\
      if sorter is not None and sorter.use_HEX else ([],il) 

    print('MOVE INPUT TO sorter if needed')
    if sorter is not None  and sorter.use_sum_correction :
        for x,y in sum_corrector_U : sorter.signal_corrector.sum_corrector_U.set_point(x,y)
        for x,y in sum_corrector_V : sorter.signal_corrector.sum_corrector_V.set_point(x,y)
        for x,y in sum_corrector_W : sorter.signal_corrector.sum_corrector_W.set_point(x,y)

    if sorter is not None  and sorter.use_pos_correction :
        for x,y in pos_corrector_U : sorter.signal_corrector.pos_corrector_U.set_point(x,y)
        for x,y in pos_corrector_V : sorter.signal_corrector.pos_corrector_V.set_point(x,y)
        for x,y in pos_corrector_W : sorter.signal_corrector.pos_corrector_W.set_point(x,y)

    return True

#----------
#----------
#----------
#----------

if __name__ == "__main__" :

    import psana.pyalgos.generic.Utils as gu

    class sorter_proxy() :
        def __init__(self) :
            pass


    def text_data(fname, do_print=False,\
                  cdir='/reg/neh/home4/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/hexanode/examples/') :
        path = cdir + fname
        if do_print : print('data file: %s' % path)        
        data = gu.load_textfile(path, verb=True)
        if do_print : print(data)
        return data


    def test_load_config_pars() :
        txt = text_data('configuration_quad.txt', do_print=True)
        load_config_pars(txt, sorter=None)


    def test_load_calibration_tables() :
        txt = text_data('calibration_table_data.txt', do_print=True)
        load_calibration_tables(txt, sorter=None)


    #print('%s\n%s'%(50*'_', USAGE))

    tname = sys.argv[1] if len(sys.argv) > 1 else '1'

    print(50*'_', '\nTest %s' % tname)
    if   tname == '1' : test_load_config_pars()
    elif tname == '2' : test_load_calibration_tables()
    else : print('Undefined test %s' % tname)

    #print('End of Test %s' % tname)
    #=====================
    sys.exit('End of Test %s' % tname)
    #=====================

#----------
