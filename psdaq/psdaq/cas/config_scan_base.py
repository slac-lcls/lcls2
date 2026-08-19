import sys
import logging
import threading
from psdaq.control.ControlDef import ControlDef, MyFloatPv, MyStringPv
from psdaq.control.DaqControl import DaqControl
from psdaq.control.ConfigScan import ConfigScan
import argparse
import json
import numpy as np
import importlib.util
from pathlib import Path
#
#  Specialization for ePix in MFX:
#    Defaults for platform, collection host, groups
#


hutch_cnf = {'tmo':'tmo_sc.py',
             'rix':'rix.py',
             'mfx':'mfx.py',
             'asc':'asc_epixuhr.py',
             'ued':'ued.py',
             'xpp':'xpp_main.py',
            }

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
    
        add_argument('-p', type=int, choices=range(0, 8), default=None, help='platform, overrides cnf and hutch')
        add_argument('-C', metavar='COLLECT_HOST', default=None,
                     help='collection host, overrides cnf and hutch')
        add_argument('--hutch' , type=str, default=None, help='hutch defines default cnf file for platform and collect_host (shortcut for -p,-C)')
        add_argument('--cnf' , type=str, default=None, help='provide cnf file used for hutch config (shortcut for -p,-C), overrides hutch')
        add_argument('-t', type=int, metavar='TIMEOUT', default=10000,
                     help='timeout msec (default 10000)')
        add_argument('-g', type=int, metavar='GROUP_MASK', help='bit mask of readout groups (default 1<<plaform)')
        add_argument('--detname', metavar='DETNAME', default='scan', help="detector name")
        add_argument('--scantype', metavar='SCANTYPE', default='scan', help="scan type")
        add_argument('--config', metavar='ALIAS', type=str, default=None, help='configuration alias (e.g. BEAM)')
        add_argument('--events', type=int, default=2000, help='events per step (default 2000)')
        add_argument('--record', type=int, choices=range(0, 2), default=None, help='recording flag')
        add_argument('--nprocs', type=int, required=False, help='Number of drp segments/processes for the detector `detname`')
        add_argument('--run_type', type=str, required=False, help='Specify a run `type`, e.g. `DARK`.')
        parser.add_argument('-v', action='store_true', help='be verbose')

        for a in userargs:
            parser.add_argument(a[0],**a[1])

        args = parser.parse_args()
        # Introducing new methods to load platform and collect_host from cnf files, 
        # it searches for default cnf files based on hutch name. This allows for more 
        # flexible configuration management.
        # -p and -C command line arguments will override any cnf or hutch settings, providing 
        # the user with the ability to specify these values directly if needed.
        # --cnf command overrides --hutch, allowing users to specify a custom configuration file (with full path)
        # --hutch will look for the operator's (defined by hutch) home directory and search for the default cnf file (defined in hutch_cnf)
        # in the expected locations: /scripts, /daq, and /daq/scripts.
        
        p = None
        C = None
        
        # asc does not have an opr account, so we use det instead. 
        if args.hutch == 'asc':
            opr='detopr'
        else:
            opr=f'{args.hutch}opr'
            
        # Establish in which system the script is running (sdf or cds) and set the home directory accordingly. 
        # This is important for locating the configuration files in the expected locations.
        if 'sdf' in str(Path.home()):
            home = f'/sdf/home/{opr[0]}/{opr}'
            basefolder='sdf'
        else:
            home = f'/cds/home/opr/{opr}'
            basefolder='cds'
        # Search for default cnf files per hutch in the expected locations: /scripts, /daq, and /daq/scripts        
        
        if args.hutch is not None :# and args.cnf is None:
            found_file=[]    
            if args.hutch in hutch_cnf:
                #logging.warning("Hutch option detected, this option has been deprecated, please use --cnf <cnf_file>or -p -C instead. ")
                for folder in [f'{home}/scripts', f'{home}/daq', f'{home}/daq/scripts/']:
                    file_name = f"{folder}/{hutch_cnf[args.hutch]}"
                    # Verify it exists and is a file (not a folder with that name)
                    if Path(file_name).is_file():
                        found_file.append(file_name)
                print(f" Suggested default cnf file(s) for hutch {args.hutch} in {basefolder}: {found_file}")
                # if no file is found, or if multiple files are found, and neither -p nor -C are provided, log an error and exit. 
                # This ensures that the user is aware of the issue and can take corrective action.
                # if len(found_file) == 0 :
                #     if args.p is None or args.C is None:
                #         logging.error(f"Config file for hutch {args.hutch} not found in expected locations")
                #         logging.error(f"--hutch searches for cnf files in {home}/scripts, {home}/daq, and {home}/daq/scripts folders")
                #         logging.error("please provide -p and -C, or --cnf full_path/file name if file not available in home.")
                #         exit(1)
                # elif len(found_file) > 1:
                #     if args.p is None or args.C is None:
                #         logging.error(f"Multiple config files found for hutch {args.hutch}: {found_file}")
                #         logging.error("Please remove duplicate and try again; check in home for /scripts, /daq, and /daq/scripts folders.")
                #         exit(1)
                # p, C = self.load_specific_variables_fromcnf(found_file[0], ['platform', 'collect_host']).values()
                # p = int(p)
                # print(f"Using hutch config for {args.hutch} from cnf file {found_file}")
            else:
                logging.warning(f"Hutch {args.hutch} not recognized,")
                #exit(1)
            
        # Use cnf file to get platform and collect_host if provided 
        # Check for human error in the cnf file name, if it doesn't end with .py, append it.
        
        if args.p is not None and args.C is not None:
            p=args.p
            C=args.C
            print(f"Using platform p and collect_host C from command line")
        else:
            if args.cnf is not None:
                if not args.cnf.endswith('.py'):
                    args.cnf += '.py'
                if Path(args.cnf).is_file():
    #                print("Found cnf file: {args.cnf}")
                    p, C = self.load_specific_variables_fromcnf(args.cnf, ['platform', 'collect_host']).values()
                    p = int(p)
                    print(f"Using p and C config from cnf file {args.cnf}")
                else:
                    logging.error(f"cnf file: {args.cnf} not found, please provide full path or check spelling")            
                    exit(1)
                    
            # Use command line arguments to override any cnf or hutch settings if provided.            
            if args.p is not None:
                p=args.p
                print(f"Using platform p from command line")
            if args.C is not None:
                C=args.C
                print(f"Using collect_host C from command line")
       
       # Validate that both platform and collect_host are specified, either through command line arguments, hutch configuration, or cnf file. 
       # If not, log an error and exit.
        if C is None or p is None:
            if C is None :
                logging.error('C is not specified')
            if p is None :
                logging.error('p is not specified')
            logging.error('please use -p <platform> and -C <COLLECT_HOST>, or --cnf <cnf_file>')
            exit(1) 
        
        # Assign the final values of platform and collect_host to the args object for use in the rest of the program.
        args.p = p
        args.C = C
                                              
        print(f"Final configuration: platform={args.p}, collect_host={args.C}")                                              
        if args.events < 1:
            logging.error('readout count (--events) must be >= 1')
            exit(1)
            
        # configure logging handlers
        if args.v:
            level=logging.DEBUG
        else:
            level=logging.WARNING
        logging.basicConfig(level=level)
        logging.info('logging initialized')
        self.args = args
        
    def load_specific_variables_fromcnf(self, filepath: str, variables: list[str]) -> dict:
            """
            Load specific variables from a .py file, discarding everything else.
            
            Args:
                filepath: Path to the .py file
                variables: List of variable names to extract
            
            Returns:
                Dictionary with only the requested variables
            """
            spec = importlib.util.spec_from_file_location("_temp_module", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            result = {}
            missing = []
            
            for var in variables:
                if hasattr(module, var):
                    result[var] = getattr(module, var)
                else:
                    missing.append(var)
            
            # Clean up — don't leave the module in sys.modules
            sys.modules.pop("_temp_module", None)
            
            if missing:
                print(f"Error: variables not found: {missing}")
                exit(1)
            return result
        
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

        # get initial DAQ state
        daqState = control.getState()
        logging.info('initial state: %s' % daqState)
        if daqState == 'error':
            sys.exit(1)

        # get the step group from the detector name
        platform = control.getPlatform()
        step_group = None
        group_mask = 0
        if 'drp' not in platform:
            sys.exit('Error: No DRP found - Have you selected DAQ components?')
        for v in platform['drp'].values():
            if v['active']==1:
                group_mask |= 1<<(v['det_info']['readout'])
                # For multi-segment/process detector args.detname may not have
                # segment number, so just check that it is in the alias instead
                # of equality
                if args.detname in v['proc_info']['alias']:
                    step_group = v['det_info']['readout']

        if step_group is None:
            step_group = args.p

        # optionally set BEAM or NOBEAM
        if args.config is not None:
            # config alias request
            rv = control.setConfig(args.config)
            if rv is not None:
                logging.error('%s' % rv)

        # Set state to a valid state for changing record setting
        control.setState("connected")
        while control.getState() != "connected": ...

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
                          "group_mask"   : group_mask,
                          'step_keys'    : keys,
                          "step_group"   : step_group, 
                          "readout_count": args.events, 
                          }  # we should have a separate group param

        enable_dict = {'group_mask'   : group_mask,
                       'step_group'   : step_group,
                       "readout_count": args.events, 
                       }

        keys_dict = {   "configure": configure_dict,
                        "enable":    enable_dict,
                        }
        
        for step in steps():
            # update
            # step value is defined from teh yield in the script:
            # yield (d, float(step), json.dumps(metad)) 
            
            d = step[0]
            nstep = step[1]
            metad = json.loads(step[2])
            
            if "events" in metad.keys(): 
                configure_dict["readout_count"] = metad["events"]
                enable_dict['readout_count']    = metad["events"]
                print(f"Number of events: {metad['events']}")
            # config scan setup
                keys_dict = {   "configure": configure_dict,
                                "enable":    enable_dict}
            
            scan.update(value=nstep)

            my_step_data = {}
            for motor in scan.getMotors():
                my_step_data.update({motor.name: motor.position})
                my_step_data.update({'step_docstring': json.dumps(metad)})

            data["motors"] = my_step_data

            beginStepBlock = scan.getBlock(transition="BeginStep", data=data)
            values_dict = \
                          {"beginstep": {"step_values":        d,
                                         "ShapesDataBlockHex": beginStepBlock}}
            # trigger
            scan.trigger(phase1Info = {**keys_dict, **values_dict})

        # -- end script ----------------------------------------------------------

        scan.unstage()

        scan.push_socket.send_string('shutdown') #shutdown the daq communicator thread
        scan.comm_thread.join()

        if args.record is not None:
            # Recording was requested, now we turn it off
            if args.record == 1:
                print("Setting record back to False")
                rv = control.setRecord(False)
