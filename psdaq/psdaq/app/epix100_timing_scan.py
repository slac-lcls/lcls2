import sys
import logging
import threading
from psdaq.control.ControlDef import ControlDef, MyFloatPv, MyStringPv
from psdaq.control.DaqControl import DaqControl
from psdaq.control.ConfigScan import ConfigScan
import argparse
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=4,
                        help='platform (default 4)')
    parser.add_argument('-C', metavar='COLLECT_HOST', default='drp-neh-cmp014',
                        help='collection host (default drp-neh-cmp014)')
    parser.add_argument('-t', type=int, metavar='TIMEOUT', default=20000,
                        help='timeout msec (default 20000)')
    parser.add_argument('-g', type=int, default=1<<4, metavar='GROUP_MASK', help='bit mask of readout groups (default 1<<plaform)')
    parser.add_argument('--config', metavar='ALIAS', default='BEAM', help='configuration alias (e.g. BEAM)')
    parser.add_argument('--detname', default='epix100_0', help="detector name (default 'epix100_0')")
    parser.add_argument('--scantype', default='timing', help="scan type (default 'timing')")
    parser.add_argument('-v', action='store_true', help='be verbose')

    parser.add_argument('--events', type=int, default=100, help='events per step (default 100)')
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

    keys = [f'{args.detname}:user.start_ns']

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
    # cpo: even though this is a config scan, I think the PV
    # is used to have an easy way to inject step_value/step_docstring
    # into the scan
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
        for start_ns in [88000,89000,90000]:
            yield int(start_ns)

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
          {"beginstep": {"step_values":        {f'{args.detname}:user.start_ns': step},
                         "ShapesDataBlockHex": beginStepBlock}}
        # trigger
        scan.trigger(phase1Info = {**keys_dict, **values_dict})

    # -- end script ----------------------------------------------------------

    scan.unstage()

    scan.push_socket.send_string('shutdown') #shutdown the daq communicator thread
    scan.comm_thread.join()


if __name__ == '__main__':
    main()
