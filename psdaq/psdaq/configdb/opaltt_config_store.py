from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import psdaq.configdb.opal_config_store as opal
import sys
import IPython
import argparse

def opaltt_cdict():

    top = opal.opal_cdict()

    #  append to the help string
    help_str = top.get("help:RO")
    help_str += "\nfex.eventcodes.beam  : beam present  = AND(.incl) and not OR(.excl)"
    help_str += "\nfex.eventcodes.laser : laser present = AND(.incl) and not OR(.excl)"
    help_str += "\nfex.roi              : inclusive columns (x) and rows (y)"
    help_str += "\nfex.signal.minvalue  : minimum signal (ADU) to report valid value"
    help_str += "\nfex..convergence     : rolling average timescale (1/N); 0 to disable correction"
    help_str += "\nfex.prescale..       : record 1/N events; 0 to disable recording"
    help_str += "\nfex.fir_weights      : edge-finding FIR constants"
    help_str += "\nfex.calib_poly       : poly coefficients for edge to time(ps)"
    top.set("help:RO", help_str, 'CHARSTR')

    #  append new fields
    top.set("fex.enable",    1, 'UINT8')

    #  assume mode is nobeam on separate events (vs nobeam in separate roi) 
    top.set("fex.eventcodes.beam.incl" , [140], 'UINT8') # beam present = AND beam.incl NOR beam.excl
    top.set("fex.eventcodes.beam.excl" , [162], 'UINT8') 
    top.set("fex.eventcodes.laser.incl", [40], 'UINT8') # laser present = AND laser.incl NOR laser.excl
    top.set("fex.eventcodes.laser.excl", [91], 'UINT8') 

    top.set("fex.sig.roi.x0",    0, 'UINT32')
    top.set("fex.sig.roi.y0",  300, 'UINT32')
    top.set("fex.sig.roi.x1", 1023, 'UINT32')
    top.set("fex.sig.roi.y1",  449, 'UINT32')
    top.set("fex.ref.enable",    0, 'UINT8')
    top.set("fex.ref.roi.x0",    0, 'UINT32')
    top.set("fex.ref.roi.y0",    0, 'UINT32')
    top.set("fex.ref.roi.x1", 1023, 'UINT32')
    top.set("fex.ref.roi.y1",  149, 'UINT32')
    top.set("fex.sb.enable" ,    1, 'UINT8')
    top.set("fex.sb.roi.x0",     0, 'UINT32')
    top.set("fex.sb.roi.y0",     0, 'UINT32')
    top.set("fex.sb.roi.x1",  1023, 'UINT32')
    top.set("fex.sb.roi.y1",   149, 'UINT32')

#    top.define_enum('boolEnum', {'False':0, 'True':1})
#    top.set("fex.subtractAndNormalize" 1, 'boolEnum')

    top.define_enum('axisEnum', {'X':0, 'Y':1})
    top.set("fex.project.axis"       ,  0, 'axisEnum')
    top.set("fex.project.minvalue"   ,  0, 'UINT32')
    top.set("fex.ref.convergence" ,  1.00, 'DOUBLE') # IIR with timescale = 1/N, 0 to disable
    top.set("fex.sb.convergence"  ,  0.05, 'DOUBLE') # IIR with timescale = 1/N, 0 to disable

    top.set("fex.prescale.image"      , 1, 'UINT32') # 0=disable
    top.set("fex.prescale.projections", 1, 'UINT32') # 0=disable

    top.define_enum('recordEnum', {'None':0, 'Projection':1, 'Image':2})
    top.set("fex.ref.record"    ,  0, 'recordEnum')

    weights = [-0.007792, -0.009892, -0.010962, -0.011497, -0.012112, -0.012101, -0.012494, -0.011907, -0.012426, -0.012944, -0.014263, -0.015331, -0.015915, -0.016482, -0.017422, -0.017410, -0.017621, -0.017650, -0.018294, -0.017623, -0.017679, -0.018247, -0.019228, -0.018376, -0.017517, -0.017561, -0.018248, -0.018338, -0.019200, -0.020193, -0.020105, -0.019915, -0.019945, -0.019453, -0.019106, -0.019508, -0.019146, -0.019066, -0.018466, -0.018704, -0.017437, -0.017540, -0.017206, -0.017021, -0.015948, -0.015219, -0.015937, -0.017049, -0.017161, -0.017260, -0.017805, -0.017646, -0.017477, -0.017698, -0.017459, -0.016523, -0.016152, -0.016095, -0.016825, -0.016119, -0.015348, -0.014191, -0.013250, -0.012839, -0.013568, -0.013127, -0.012936, -0.012412, -0.010400, -0.009663, -0.009195, -0.007919, -0.007892, -0.007097, -0.006648, -0.007520, -0.007855, -0.006912, -0.006410, -0.006255, -0.006033, -0.005148, -0.005202, -0.005424, -0.005026, -0.004917, -0.004676, -0.004847, -0.003644, -0.003399, -0.003085, -0.002936, -0.001964, -0.002192, -0.001741, -0.001722, -0.001276, -0.001391, -0.001434, -0.000984, -0.000178, 0.000372, 0.000876, 0.001563, 0.002183, 0.001668, 0.002241, 0.002896, 0.003262, 0.004451, 0.004892, 0.005819, 0.007591, 0.008409, 0.008896, 0.009513, 0.009288, 0.010103, 0.011191, 0.012232, 0.013822, 0.014685, 0.014454, 0.014597, 0.015290, 0.016948, 0.017363, 0.017374, 0.017347, 0.017966, 0.018080, 0.017962, 0.017797, 0.016955, 0.016971, 0.017602, 0.018404, 0.018555, 0.019103, 0.018996, 0.020581, 0.020389, 0.020994, 0.021038, 0.021340, 0.021512, 0.021724, 0.022064, 0.022348, 0.022657, 0.022909, 0.021844, 0.021123, 0.021566, 0.020816, 0.020502, 0.021025, 0.022016, 0.021695, 0.021063, 0.020254, 0.020797, 0.020017, 0.019454, 0.019175, 0.019282, 0.018357, 0.017892, 0.017803, 0.016783, 0.015552, 0.015116, 0.014447, 0.014949, 0.015110, 0.015786, 0.015646, 0.016084, 0.015854, 0.014134, ]
    top.set("fex.fir_weights", weights, 'DOUBLE')

    top.define_enum('boolEnum', {'False':0, 'True':1})
    top.set("fex.invert_weights", 0, 'boolEnum')

    calib = [1.138000, -0.002390, 0.000000, ]
    top.set("fex.calib_poly" , calib, 'DOUBLE') # time(ps) = sum( calib_poly[i]*edge^i )

    return top

if __name__ == "__main__":
    args = cdb.createArgs().args

    args.name = 'tmoopal2'
    args.segm = 0
    args.id = 'opal_serial1235'
    args.alias = 'BEAM'
    args.prod = True
    args.inst = 'tmo'
    args.user = 'tmoopr'
    args.password = 'pcds'

    create = True
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

    db   = 'configdb' if args.prod else 'devconfigdb'
    url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

    mycdb = cdb.configdb(url, args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config('opal')

    top = opaltt_cdict()
    top.setInfo('opal', args.name, args.segm, args.id, 'No comment')

    mycdb.modify_device(args.alias, top)
