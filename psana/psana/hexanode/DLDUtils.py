#!/usr/bin/env python
"""
Module :py:class:`DLDUtils` 
=========================================================================

    from psana.hexanode.DLDUtils import load_config_pars, load_calibration_tables, text_data

    txt = text_data(file_name.txt)

    status, command, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y=\
    load_config_pars(txt, sorter, **kwargs) # if in kwargs command=1, it is taken from file

    status = load_calibration_tables(txt, sorter)

Created on 2019-12-12 by Mikhail Dubrovin
"""
#----------

USAGE = 'Run example: python .../psana/hexanode/examples/ex-25-quad-proc-data.py'

#----------

import logging
logger = logging.getLogger(__name__)

import sys
import psana.pyalgos.generic.Utils as gu

#----------

def load_int(line) :
    return int(line.strip().split()[0]) # .atoi()


def load_double(line) :
    return float(line.strip().split()[0]) # .atof()


def load_double_couple(line) :
    flds = line.strip().split()
    return float(flds[0]), float(flds[1])


def load_config_pars(txt_config, sorter=None, **kwargs) :
    """in stead of py_read_config_file"""

    if not txt_config : return False, None, None, None, None, None, None, None

    lines = txt_config.split('\n')
    lins = [l for l in lines if l] # discard empty lines

    #for line in lins : print('==:', line)

    COMMAND = kwargs.get('command',1)
    command = load_int(lins[0])

    # CLI 'command' argument has higher priority then in configuration file
    if COMMAND!=1 : command = COMMNAD 

    if command == -1 : return False, None, None, None, None, None, None, None

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

    msg = 'Configuration constants loaded:'\
        + '\n  command:            %d' % command\
        + '\n  use_hex:            %s' % use_hex\
        + '\n  common_start_mode:  %s' % common_start_mode\
        + '\n  channels u1,cu2,cv1,cv2,cw1,cw2,cmcp: %2d %2d %2d %2d %2d %2d %2d'%(cu1,cu2,cv1,cv2,cw1,cw2,cmcp)\
        + '\n  use_mcp:            %s' % use_mcp\
        + '\n  offset_sum_u:    %6.1f' % offset_sum_u\
        + '\n  offset_sum_v:    %6.1f' % offset_sum_v\
        + '\n  offset_sum_w:    %6.1f' % offset_sum_w\
        + '\n  pos_offset_x:    %6.1f' % pos_offset_x\
        + '\n  pos_offset_y:    %6.1f' % pos_offset_y\
        + '\n  uncorrected_time_sum_half_width_u: %4.1f' % uncorrected_time_sum_half_width_u\
        + '\n  uncorrected_time_sum_half_width_v: %4.1f' % uncorrected_time_sum_half_width_v\
        + '\n  uncorrected_time_sum_half_width_w: %4.1f' % uncorrected_time_sum_half_width_w\
        + '\n  fu:              %6.3f' % fu\
        + '\n  fv:              %6.3f' % fv\
        + '\n  fw:              %6.3f' % fw\
        + '\n  w_offset:        %6.1f' % w_offset\
        + '\n  runtime_u:       %6.1f' % runtime_u\
        + '\n  runtime_v:       %6.1f' % runtime_v\
        + '\n  runtime_w:       %6.1f' % runtime_w\
        + '\n  mcp_radius:      %6.2f' % mcp_radius\
        + '\n  dead_time_anode: %6.3f' % dead_time_anode\
        + '\n  dead_time_mcp:   %6.3f' % dead_time_mcp\
        + '\n  use_sum_correction: %s' % use_sum_correction\
        + '\n  use_pos_correction: %s' % use_pos_correction\
        + '\n  check_input:        %d' % check_input

    logger.info(msg)

    if check_input != 88888 : 
        logger.warning("Configuration file was not correctly read.")
        # close file
        # delete sorter
        sys.exit("Configuration file was not correctly read.")

    if sorter is not None:
        sorter.use_hex = use_hex

        # pass values to sorter sorter
        sorter.cu1 = cu1
        sorter.cu2 = cu2
        sorter.cv1 = cv1
        sorter.cv2 = cv2
        sorter.cw1 = cw1
        sorter.cw2 = cw2
        sorter.cmcp = cmcp
        sorter.use_mcp = use_mcp

        sorter.uncorrected_time_sum_half_width_u = uncorrected_time_sum_half_width_u
        sorter.uncorrected_time_sum_half_width_v = uncorrected_time_sum_half_width_v
        sorter.uncorrected_time_sum_half_width_w = uncorrected_time_sum_half_width_w

        sorter.fu = fu
        sorter.fv = fv
        sorter.fw = fw

        sorter.runtime_u = runtime_u
        sorter.runtime_v = runtime_v
        sorter.runtime_w = runtime_w
        sorter.mcp_radius = mcp_radius

        sorter.dead_time_anode = dead_time_anode
        sorter.dead_time_mcp = dead_time_mcp
        sorter.use_sum_correction = use_sum_correction
        sorter.use_pos_correction = use_pos_correction

    return True, command, offset_sum_u, offset_sum_v, offset_sum_w,\
           w_offset, pos_offset_x, pos_offset_y

#----------

def load_calibration_group(lins, il, cmt='') :
    points = load_int(lins[il])
    logger.info('DLDUtils.load_calibration_group - %2d points for %s' % (points, cmt))

    list_of_pairs = []
    for i in range(points) :
        il += 1
        x,y = load_double_couple(lins[il])
        #print('l:%2d x=%8.3f y=%8.3f' % (i,x,y))
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
        if sorter is not None and sorter.use_hex else ([],il)

    pos_corrector_U, il = load_calibration_group(lins, il, cmt='pos_corrector_U')
    pos_corrector_V, il = load_calibration_group(lins, il, cmt='pos_corrector_V')
    pos_corrector_W, il = load_calibration_group(lins, il, cmt='pos_corrector_W')\
        if sorter is not None and sorter.use_hex else ([],il) 

    #logger.debug('MOVE INPUT TO sorter if needed')
    if sorter is not None  and sorter.use_sum_correction :
        for x,y in sum_corrector_U : sorter.signal_corrector.sum_corrector_U_set_point(x,y)
        for x,y in sum_corrector_V : sorter.signal_corrector.sum_corrector_V_set_point(x,y)
        for x,y in sum_corrector_W : sorter.signal_corrector.sum_corrector_W_set_point(x,y)

    if sorter is not None  and sorter.use_pos_correction :
        for x,y in pos_corrector_U : sorter.signal_corrector.pos_corrector_U_set_point(x,y)
        for x,y in pos_corrector_V : sorter.signal_corrector.pos_corrector_V_set_point(x,y)
        for x,y in pos_corrector_W : sorter.signal_corrector.pos_corrector_W_set_point(x,y)

    return True

#----------

def text_data(fname) :
    logger.info('DLDUtils.text_data from file') #: %s' % fname)        
    data = gu.load_textfile(fname, verb=True)
    logger.debug(data)
    return data

#----------
#----------
#----------

if __name__ == "__main__" :

    #fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s' # '%(message)s'
    fmt='%(levelname)s: %(message)s'
    logging.basicConfig(format=fmt, datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG) #.INFO)

    CDIR ='/reg/neh/home4/dubrovin/LCLS/con-lcls2/lcls2/psana/psana/hexanode/examples/'

    class sorter_proxy() :
        def __init__(self) :
            pass


    def test_load_config_pars() :
        txt = text_data(CDIR + 'configuration_quad.txt')
        status, command, offset_sum_u, offset_sum_v, offset_sum_w, w_offset, pos_offset_x, pos_offset_y=\
        load_config_pars(txt, sorter=None)


    def test_load_calibration_tables() :
        txt = text_data(CDIR + 'calibration_table_data.txt')
        status = load_calibration_tables(txt, sorter=None)


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
