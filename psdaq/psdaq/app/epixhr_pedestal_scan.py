import sys
import logging
import threading
from psdaq.control.ControlDef import ControlDef, MyFloatPv, MyStringPv
from psdaq.control.DaqControl import DaqControl
from psdaq.control.ConfigScan import ConfigScan
import argparse
import json
import numpy as np

#
#  Use the AMI MeanVsScan plot
#    Not that its binning has an error
#    The first,last bin will not be filled; 
#    The other bins need to be shifted
#    So, for (100,1800100,90000) the binning should be (21,-89800,1800200)
#    (Valid plot points will be 200,90200,...,1710200 inclusive)
#
#  Trouble when XPM L0Delay is < 70
#    KCU doesn't send all triggers
#    Minimum trigger delay is then 77 us
#    (likely solved by l2si-core trigfifo-fix)
#
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=2,
                        help='platform (default 2)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='drp-neh-ctl001',
                        help='collection host (default drp-neh-ctl001)')
    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=20000,
                        help='timeout msec (default 20000)')
    parser.add_argument('-g', type=int, default=6, metavar='GROUP_MASK', help='bit mask of readout groups (default 1<<plaform)')
    parser.add_argument('--config', metavar='ALIAS', default='BEAM', help='configuration alias (e.g. BEAM)')
    parser.add_argument('--detname', default='epixhr_0', help="detector name (default 'scan')")
    parser.add_argument('--scantype', default='pedestal', help="scan type (default 'scan')")
    parser.add_argument('-v', action='store_true', help='be verbose')

    parser.add_argument('--events', type=int, default=2000, help='events per step (default 2000)')
    parser.add_argument('--record', type=int, choices=range(0, 2), help='recording flag')

    args = parser.parse_args()

    if args.g is not None:
        if args.g < 1 or args.g > 255:
            parser.error('readout group mask (-g) must be 1-255')
        group_mask = args.g
    else:
        group_mask = 1 << args.p

    if args.events < 1:
        parser.error('readout count (--events) must be >= 1')

    keys = [f'{args.detname}:user.gain_mode']

    # instantiate DaqControl object
    control = DaqControl(host=args.C, platform=args.p, timeout=args.t)

    try:
        instrument = control.getInstrument()
    except KeyboardInterrupt:
        instrument = None

    if instrument is None:
        sys.exit('Error: failed to read instrument name (check -C <COLLECT_HOST>)')

    # configure logging handlers
    if args.v:
        level=logging.DEBUG
    else:
        level=logging.WARNING
    logging.basicConfig(level=level)
    logging.info('logging initialized')

    # get initial DAQ state
    daqState = control.getState()
    logging.info('initial state: %s' % daqState)
    if daqState == 'error':
        sys.exit(1)

    # optionally set BEAM or NOBEAM
    if args.config is not None:
        # config alias request
        rv = control.setConfig(args.config)
        if rv is not None:
            logging.error('%s' % rv)

    if args.record is not None:
        # recording flag request
        if args.record == 0:
            rv = control.setRecord(False)
        else:
            rv = control.setRecord(True)
        if rv is not None:
            print('Error: %s' % rv)

    # instantiate ConfigScan
    scan = ConfigScan(control, daqState=daqState, args=args)

    scan.stage()

    # -- begin script --------------------------------------------------------

    # PV scan setup
    motors = [MyFloatPv(ControlDef.STEP_VALUE)]
    scan.configure(motors = motors)

    my_config_data = {}
    for motor in scan.getMotors():
        my_config_data.update({motor.name: motor.position})
        # derive step_docstring from step_value
        if motor.name == ControlDef.STEP_VALUE:
            docstring = f'{{"detname": "{args.detname}", "scantype": "{args.scantype}", "step": {motor.position}}}'
            my_config_data.update({'step_docstring': docstring})

    data = {
      "motors":           my_config_data,
      "timestamp":        0,
      "detname":          "scan",
      "dettype":          "scan",
      "scantype":         args.scantype,
      "serial_number":    "1234",
      "alg_name":         "raw",
      "alg_version":      [1,0,0]
    }

    configureBlock = scan.getBlock(transition="Configure", data=data)

    # config scan setup
    keys_dict = {"configure": {"step_keys":     keys,
                               "NamesBlockHex": configureBlock},
                 "enable":    {"readout_count": args.events,
                               "group_mask":    group_mask}}
    # scan loop
    def steps():
        for gain in range(5):
            yield int(gain)

    for step in steps():
        # update
        scan.update(value=scan.step_count())

        my_step_data = {}
        for motor in scan.getMotors():
            my_step_data.update({motor.name: motor.position})
            # derive step_docstring from step_value
            if motor.name == ControlDef.STEP_VALUE:
                #  Need an integer for "step" value.  Config scan analysis uses as an array index.
                docstring = f'{{"detname": "{args.detname}", "scantype": "{args.scantype}", "step": {int(motor.position)}}}'
                my_step_data.update({'step_docstring': docstring})

        data["motors"] = my_step_data

        beginStepBlock = scan.getBlock(transition="BeginStep", data=data)
        values_dict = \
          {"beginstep": {"step_values":        {f'{args.detname}:user.gain_mode': step},
                         "ShapesDataBlockHex": beginStepBlock}}
        # trigger
        scan.trigger(phase1Info = {**keys_dict, **values_dict})

    # -- end script ----------------------------------------------------------

    scan.unstage()

    scan.push_socket.send_string('shutdown') #shutdown the daq communicator thread
    scan.comm_thread.join()


if __name__ == '__main__':
    main()
