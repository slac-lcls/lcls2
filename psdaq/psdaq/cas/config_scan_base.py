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
#  Specialization for ePix in MFX:
#    Defaults for platform, collection host, groups
#
hutch_def = {'tmo':(0,'drp-srcf-mon001'),
             'rix':(0,'drp-srcf-mon002'),
             'ued':(0,'drp-ued-cmp002'),
             'asc':(2,'drp-det-cmp001')}

class ConfigScanBase(object):
    def __init__(self, userargs=[], defargs={}):

        #
        #  add arguments with defaults given by caller
        #
        parser = argparse.ArgumentParser()
        def add_argument(arg,metavar='',default=None,required=False,help='',**kwargs):
            if arg in defargs:
                default = defargs[arg]
                required = False
            parser.add_argument(arg,default=default,metavar=metavar,required=required,help=help,**kwargs)

        add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
        add_argument('-C', metavar='COLLECT_HOST', default='localhost',
                     help='collection host (default localhost)')
        add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                     help='timeout msec (default 10000)')
        add_argument('-c', type=int, metavar='READOUT_COUNT', default=1, help='# of events to aquire at each step (default 1)')
        add_argument('-g', type=int, metavar='GROUP_MASK', help='bit mask of readout groups (default 1<<plaform)')
        add_argument('--detname', metavar='DETNAME', default='scan', help="detector name")
        add_argument('--scantype', metavar='SCANTYPE', default='scan', help="scan type")
        add_argument('--config', metavar='ALIAS', type=str, default=None, help='configuration alias (e.g. BEAM)')
        add_argument('--events', type=int, default=2000, help='events per step (default 2000)')
        add_argument('--record', type=int, choices=range(0, 2), default=None, help='recording flag')
        add_argument('--hutch' , type=str, default=None, help='hutch (shortcut for -p,-C)')
        parser.add_argument('-v', action='store_true', help='be verbose')

        for a in userargs:
            parser.add_argument(a[0],**a[1])

        args = parser.parse_args()

        if args.hutch and args.hutch in hutch_def:
            args.p, args.C = hutch_def[args.hutch]

        if args.events < 1:
            parser.error('readout count (--events) must be >= 1')

        self.args = args

    def run(self,keys,steps):

        args = self.args

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

        # get the step group from the detector name
        platform = control.getPlatform()
        step_group = None
        group_mask = 0
        for v in platform['drp'].values():
            if v['active']==1:
                group_mask |= 1<<(v['det_info']['readout'])
                if v['proc_info']['alias'] == args.detname:
                    step_group = v['det_info']['readout']

        if step_group is None:
            step_group = args.p

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

        configure_dict = {"NamesBlockHex": configureBlock,
                          "readout_count": args.events,
                          "group_mask"   : group_mask,
                          'step_keys'    : keys,
                          "step_group"   : step_group }  # we should have a separate group param

        enable_dict = {'readout_count': args.events,
                       'group_mask'   : group_mask,
                       'step_group'   : step_group }

        # config scan setup
        keys_dict = {"configure": configure_dict,
                     "enable":    enable_dict}

        for step in steps():
            # update
            scan.update(value=step[1])

            my_step_data = {}
            for motor in scan.getMotors():
                my_step_data.update({motor.name: motor.position})
                my_step_data.update({'step_docstring': step[2]})

            data["motors"] = my_step_data

            beginStepBlock = scan.getBlock(transition="BeginStep", data=data)
            values_dict = \
                          {"beginstep": {"step_values":        step[0],
                                         "ShapesDataBlockHex": beginStepBlock}}
            # trigger
            scan.trigger(phase1Info = {**keys_dict, **values_dict})

        # -- end script ----------------------------------------------------------

        scan.unstage()

        scan.push_socket.send_string('shutdown') #shutdown the daq communicator thread
        scan.comm_thread.join()
