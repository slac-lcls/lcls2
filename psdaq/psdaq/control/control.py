import os
import time
import copy
import socket
from datetime import datetime, timezone, timedelta
import json as oldjson
import zmq
import zmq.utils.jsonapi as json
from transitions import Machine, MachineError, State
import argparse
import requests
from requests.auth import HTTPBasicAuth
import logging
from psalg.utils.syslog import SysLog
import string
from p4p.client.thread import Context
import epics
from threading import Thread, Event, Condition
import dgramCreate as dc
from psdaq.control.ControlDef import ControlDef, create_msg, error_msg, warning_msg, step_msg, \
                                  progress_msg, fileReport_msg, front_pub_port, step_pub_port, \
                                  back_pub_port, front_rep_port, back_pull_port, fast_rep_port

report_keys = ['error', 'warning', 'fileReport']

class PvInfo:
    """PV"""
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc

    def get_name(self):
        return self.name

    def get_desc(self):
        return self.name if self.desc is None else self.desc

    def __repr__(self):
        return f'PvInfo(name={self.name} desc={self.desc})'

class RunParams:
    """Run Parameters"""
    def __init__(self, path, collection, pva):
        self.path = path
        self.collection = collection
        self.pva = pva
        self.dirname = os.path.dirname(path)
        self.fileSet = set()
        self.pvaList = []
        self.caList = []

    def updateFileSet(self, fileSet):
        logging.debug(f"RunParams updateFileSet({fileSet})")
        found = set()
        for ff in fileSet:
            try:
                with open(ff, "r") as fd:
                    for rawline in fd.readlines():
                        # remove comments and leading/trailing whitespace
                        line = rawline.split('#')[0].strip()
                        if line.startswith('<'):
                            # parse CSV
                            for rawname in line[1:].split(","):
                                rarename = rawname.strip()
                                if os.path.isabs(rarename):
                                    found.add(rarename)
                                else:
                                    found.add(self.dirname + '/' + rarename)
            except Exception as ex:
                logging.error('updateFileSet() Exception: %s' % ex)
        logging.debug(f"RunParams updateFileSet(): found={found}")
        return fileSet | found

    def updatePvSet(self):
        logging.debug("RunParams updatePvSet()")
        pvaFound = []
        caFound = []
        desc = None
        for ff in self.fileSet:
            try:
                with open(ff, "r") as fd:
                    for veryrawline in fd.readlines():
                        rawline = veryrawline.strip()
                        if rawline.startswith('<'):
                            # file include
                            desc = None
                            continue
                        if rawline.startswith('*') or rawline.startswith('#*'):
                            # PV description
                            desc = rawline.split('*')[1].strip()
                        elif len(rawline) == 0 or rawline.startswith('#'):
                            # comment
                            desc = None
                            continue
                        else:
                            # PV name
                            # PV names may be followed by a provider type (ca or pva), else ca is used
                            nnn = rawline.strip().split()
                            name = nnn[0]
                            if len(nnn) == 2 and nnn[1] == 'pva':
                                # pva...
                                pvaFound.append(PvInfo(name, desc))
                                logging.debug(f'PV: name=\'{name}\' desc=\'{desc}\' provider=\'pva\'')
                            else:
                                # ca...
                                caFound.append(PvInfo(name, desc))
                                logging.debug(f'PV: name=\'{name}\' desc=\'{desc}\' provider=\'ca\'')
            except Exception as ex:
                logging.error('updatePvSet() Exception: %s' % ex)
        logging.debug(f"RunParams updatePvSet(): pvaFound={pvaFound} caFound={caFound}")
        self.pvaList = pvaFound
        self.caList = caFound
        return

    def configure(self):
        if (self.path != '/dev/null') and not os.path.isfile(self.path):
            self.collection.report_error(f"logbook run parameter file not found: {self.path}")
        else:
            logging.info(f"logbook run parameter file: {self.path}")
            before = set([self.path])
            while True:
                self.fileSet |= before
                after = self.updateFileSet(before)
                newfound = after - self.fileSet
                if len(newfound) > 0:
                    # new files found, so recurse
                    before = newfound
                    continue
                else:
                    # no new files found, so we're done
                    break
            missingFiles = set()
            for filename in self.fileSet:
                if (filename != '/dev/null') and not os.path.isfile(filename):
                    missingFiles.add(filename)
                    self.collection.report_error(f"logbook run parameter file not found: {filename}")
            # do not modify self.fileSet while iterating over it
            self.fileSet -= missingFiles
        logging.debug(f"RunParams configure(): fileSet={self.fileSet}")
        if len(self.fileSet) > 0:
            self.updatePvSet()
        logging.debug(f"RunParams configure(): pvaList={self.pvaList} caList={self.caList}")
        self.recordedExperiments = set()    # updated in beginrun

    def unconfigure(self):
        logging.debug("RunParams unconfigure()")
        self.fileSet = set()
        self.pvaList = []
        self.caList = []

    def beginrun(self, experiment_name):
        logging.debug("mmm")
        logging.debug(f"RunParams beginrun() experiment_name={experiment_name}")
        inCount = len(self.pvaList) + len(self.caList)
        errorCount = 0
        params = {}
        param_descs = {}

        if not experiment_name in self.recordedExperiments:
            # gather PV run parameter descriptions
            for ppp in self.pvaList + self.caList:
                desc = ppp.get_desc()
                param_descs[ppp.get_name()] = desc
            self.recordedExperiments.add(experiment_name)

        logging.debug(f"RunParams: param_descs = {param_descs}")

        # gather pva run parameters
        nameList = []
        for ppp in self.pvaList:
            nameList.append(ppp.get_name())
        valueList = self.pva.pv_get(nameList)
        for name, value in zip(nameList, valueList):
            if isinstance(value, TimeoutError):
                self.collection.report_error(f"failed to read PVA PV {name}")
            else:
                params[name] = value

        # gather ca run parameters
        nameList = []
        for ppp in self.caList:
            nameList.append(ppp.get_name())
        valueList = epics.caget_many(nameList)
        for name, value in zip(nameList, valueList):
            if value is None:
                self.collection.report_error(f"failed to read CA PV {name}")
            else:
                params[name] = value

        # gather detector run parameters
        for level, item in self.collection.cmstate_levels().items():
            if level == "drp":
                for xid in item.keys():
                    try:
                        if item[xid]['active'] != 1:
                            # skip inactive detector
                            continue
                        else:
                            alias = item[xid]['proc_info']['alias']
                    except KeyError as ex:
                        logging.error('KeyError: %s' % ex)
                    else:
                        params[f"DAQ Detectors/{level}/{alias}"] = True
                        inCount += 1

        # add run parameters to logbook
        outCount = self.collection.add_run_params(experiment_name, params)
        if outCount < inCount:
            self.collection.report_error(f"{outCount} of {inCount} run parameters recorded in logbook (experiment={experiment_name})")
        else:
            logging.info(f"{outCount} run parameters recorded in logbook (experiment={experiment_name})")

        # add run parameter descriptions to logbook
        inCount = len(param_descs)
        outCount = self.collection.add_update_run_param_descriptions(experiment_name, param_descs)
        if outCount < inCount:
            self.collection.report_error(f"{outCount} of {inCount} run parameter descriptions recorded in logbook (experiment={experiment_name})")
        elif outCount > 0:
            logging.info(f"{outCount} run parameter descriptions recorded in logbook (experiment={experiment_name})")

class ControlError(Exception):
    """Base class for exceptions in this module."""
    pass

class ConfigDBError(ControlError):
    """Exception raised for ConfigDB errors.
    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message):
        self.message = message

next_dict = {
    'reset' :       { 'unallocated' : 'rollcall',
                      'allocated' :   'rollcall',
                      'connected' :   'rollcall',
                      'configured' :  'rollcall',
                      'starting' :    'rollcall',
                      'paused' :      'rollcall',
                      'running' :     'rollcall' },

    'unallocated' : { 'reset' :       'reset',
                      'allocated' :   'alloc',
                      'connected' :   'alloc',
                      'configured' :  'alloc',
                      'starting' :    'alloc',
                      'paused' :      'alloc',
                      'running' :     'alloc' },

    'allocated' :   { 'reset' :       'reset',
                      'unallocated' : 'dealloc',
                      'connected' :   'connect',
                      'configured' :  'connect',
                      'starting' :    'connect',
                      'paused' :      'connect',
                      'running' :     'connect' },

    'connected' :   { 'reset' :       'reset',
                      'unallocated' : 'disconnect',
                      'allocated' :   'disconnect',
                      'configured' :  'configure',
                      'starting' :    'configure',
                      'paused' :      'configure',
                      'running' :     'configure' },

    'configured' :  { 'reset' :       'reset',
                      'unallocated' : 'unconfigure',
                      'allocated' :   'unconfigure',
                      'connected' :   'unconfigure',
                      'starting' :    'beginrun',
                      'paused' :      'beginrun',
                      'running' :     'beginrun' },

    'starting' :    { 'reset' :       'reset',
                      'unallocated' : 'endrun',
                      'allocated' :   'endrun',
                      'connected' :   'endrun',
                      'configured' :  'endrun',
                      'paused' :      'beginstep',
                      'running' :     'beginstep' },

    'paused' :      { 'reset' :       'reset',
                      'unallocated' : 'endstep',
                      'allocated' :   'endstep',
                      'connected' :   'endstep',
                      'configured' :  'endstep',
                      'starting' :    'endstep',
                      'running' :     'enable' },

    'running' :     { 'reset' :       'reset',
                      'unallocated' : 'disable',
                      'allocated' :   'disable',
                      'connected' :   'disable',
                      'configured' :  'disable',
                      'starting' :    'disable',
                      'paused' :      'disable' }
}

# Translate drp alias to detector name
# For example: 'cam_1' -> 'cam'
def detector_name(drp_alias):
    return drp_alias.rsplit('_', 1)[0]

# Count the number of drp segments matching a detector name.
# If only_active=True, count only the active segments.
def segment_count(det_name, platform_dict, *, only_active=False):
    count = 0
    try:
        for v in platform_dict['drp'].values():
            if only_active and not (v['active'] == 1):
                # skip inactive segment
                continue
            if det_name == detector_name(v['proc_info']['alias']):
                count += 1
    except KeyError:
        pass

    return count

def timestampStr():
    current = datetime.now(timezone.utc)
    nsec = 1000 * current.microsecond
    sec = int(current.timestamp()) - POSIX_TIME_AT_EPICS_EPOCH
    return '%010d-%09d' % (sec, nsec)

def get_readout_group_mask(body):
    mask = 0
    if 'drp' in body:
        for key, node_info in body['drp'].items():
            try:
                mask |= (1 << node_info['det_info']['readout'])
            except KeyError:
                pass
    return mask

def wait_for_answers(socket, wait_time, msg_id):
    """
    Wait and return all messages from socket that match msg_id
    Parameters
    ----------
    socket: zmq socket
    wait_time: int, wait time in milliseconds
    msg_id: int or None, expected msg_id of received messages
    """
    global report_keys
    remaining = wait_time
    start = time.time()
    while socket.poll(remaining) == zmq.POLLIN:
        try:
            msg = socket.recv_json()
        except Exception as ex:
            logging.error('recv_json(): %s' % ex)
            continue
        else:
            logging.debug('recv_json(): %s' % msg)

        # handle async reports
        if msg['header']['key'] in report_keys:
            yield msg
            continue

        # if msg_id is none take the msg_id of the first message as reference
        if msg_id is None:
            msg_id = msg['header']['msg_id']

        if msg['header']['msg_id'] == msg_id:
            yield msg
        else:
            logging.error('unexpected msg_id: got %s but expected %s' %
                          (msg['header']['msg_id'], msg_id))
        remaining = max(0, int(wait_time - 1000*(time.time() - start)))

def levels_to_activedet(src):
    dst = {"activedet": {}}
    for level, item1 in src.items():
        if level == "control":
            continue    # skip
        if level not in dst["activedet"]:
            dst["activedet"][level] = {}
        for xx, item2 in item1.items():
            alias = item2["proc_info"]["alias"]
            dst["activedet"][level][alias] = {}
            if "det_info" in item2:
                dst["activedet"][level][alias]["det_info"] = item2["det_info"].copy()
            dst["activedet"][level][alias]["active"] = item2["active"]
    return dst

class DaqPVA():
    def __init__(self, *, platform, xpm_master, pv_base):
        self.platform         = platform
        self.xpm_master       = xpm_master
        self.pv_xpm_base      = pv_base + ':XPM:%d'         % xpm_master
        self.pv_xpm_part_base = pv_base + ':XPM:%d:PART:%d' % (xpm_master, platform)

        # name PVs
        self.pvListMsgHeader  = []  # filled in at alloc
        self.pvListXPM        = []  # filled in at alloc
        self.pvGroupL0Enable  = self.pv_xpm_base+':GroupL0Enable'
        self.pvGroupL0Disable = self.pv_xpm_base+':GroupL0Disable'
        self.pvGroupMsgInsert = self.pv_xpm_base+':GroupMsgInsert'
        self.pvGroupL0Reset   = self.pv_xpm_base+':GroupL0Reset'
        self.pvStepGroups     = self.pv_xpm_part_base+':StepGroups'
        self.pvStepDone       = self.pv_xpm_part_base+':StepDone'
        self.pvStepEnd        = self.pv_xpm_part_base+':StepEnd'

        # initialize EPICS context
        self.ctxt = Context('pva', nt=None)

    #
    # DaqPVA.step_groups -
    #
    # If you don't want steps, set StepGroups = 0.
    #
    def step_groups(self, *, mask):
        logging.debug("DaqPVA.step_groups(mask=%d)" % mask)
        return self.pv_put(self.pvStepGroups, mask)

    #
    # DaqPVA.pv_get -
    #
    # Return a list of PV values or exceptions.
    #
    def pv_get(self, pvList):
        if pvList is None:
            retval = []
        elif isinstance(pvList, list):
            retval = self.ctxt.get(pvList, throw=False)
        else:
            retval = self.ctxt.get([pvList], throw=False)
        logging.debug(f"DaqPVA.pv_get({pvList}) = {retval}")
        return retval

    #
    # DaqPVA.pv_put -
    #
    def pv_put(self, pvName, val):

        retval = False

        try:
            self.ctxt.put(pvName, val)
        except TimeoutError:
            logging.error("self.ctxt.put('%s', %d) timed out" % (pvName, val))
        except Exception:
            logging.error("self.ctxt.put('%s', %d) failed" % (pvName, val))
        else:
            retval = True
            logging.debug("self.ctxt.put('%s', %d)" % (pvName, val))

        return retval

    #
    # DaqPVA.monitor_StepDone
    #
    def monitor_StepDone(self, *, callback):
        return self.ctxt.monitor(self.pvStepDone, callback)


class CollectionManager():
    def __init__(self, args):
        self.platform = args.p
        self.alias = args.u
        self.config_alias = args.C  # e.g. BEAM/NOBEAM
        self.cfg_dbase = args.d
        self.trigger_config = args.t
        self.xpm_master = args.x
        self.pv_base = args.B
        self.context = zmq.Context(1)
        self.back_pull = self.context.socket(zmq.PULL)
        self.back_pub = self.context.socket(zmq.PUB)
        self.front_rep = self.context.socket(zmq.REP)
        self.fast_rep = self.context.socket(zmq.REP)
        self.front_pub = self.context.socket(zmq.PUB)
        self.back_pull.bind('tcp://*:%d' % back_pull_port(args.p))
        self.back_pub.bind('tcp://*:%d' % back_pub_port(args.p))
        self.front_rep.bind('tcp://*:%d' % front_rep_port(args.p))
        self.fast_rep.bind('tcp://*:%d' % fast_rep_port(args.p))
        self.front_pub.bind('tcp://*:%d' % front_pub_port(args.p))
        self.slow_update_rate = args.S if args.S else 0
        self.fast_reply_rate = 10           # Hz
        self.slow_update_enabled = False    # setter: self.set_slow_update_enabled()
        self.threads_exit = Event()
        self.phase2_timeout = args.T
        self.user = args.user
        self.password = args.password
        self.url = args.url
        self.experiment_name = None
        self.run_number = self.last_run_number = 0
        self.rollcall_timeout = args.rollcall_timeout
        self.bypass_activedet = False
        self.cydgram = dc.CyDgram()
        self.step_done = Event()
        self.readoutCumulative = 0

        # instantiate DaqPVA object
        self.pva = DaqPVA(platform=self.platform, xpm_master=self.xpm_master, pv_base=self.pv_base)

        # instantiate RunParams object
        self.runParams = RunParams(args.V, self, self.pva)

        if args.r:
            # active detectors file from command line
            self.activedetfilename = args.r
        else:
            # default active detectors file
            homedir = os.path.expanduser('~')
            self.activedetfilename = '%s/.psdaq/p%d.activedet.json' % (homedir, self.platform)

        if self.activedetfilename == '/dev/null':
            # active detectors file bypassed
            self.bypass_activedet = True
            logging.warning("active detectors file disabled. Default settings will be used.")
        else:
            logging.info("active detectors file: %s" % self.activedetfilename)

        # initialize fast reply thread
        self.fast_reply_thread = Thread(target=self.fast_reply_func, name='fastreply')

        if self.slow_update_rate:
            # initialize slow update thread
            self.slow_update_thread = Thread(target=self.slow_update_func, name='slowupdate')
        else:
            self.slow_update_thread = None

        # initialize stepdone thread
        self.step_done_thread = Thread(target=self.step_done_func, name='stepdone')

        # initialize poll set
        self.poller = zmq.Poller()
        self.poller.register(self.back_pull, zmq.POLLIN)
        self.poller.register(self.front_rep, zmq.POLLIN)

        # initialize fast poll set
        self.fast_poller = zmq.Poller()
        self.fast_poller.register(self.fast_rep, zmq.POLLIN)

        # initialize EPICS context
        self.ctxt = Context('pva')

        self.groups = 0     # groups bitmask
        self.cmstate = {}
        self.phase1Info = {}
        self.level_keys = {'drp', 'teb', 'meb', 'control'}

        # parse instrument_name[:station_number]
        if ':' in args.P:
            self.instrument, station_number = args.P.split(':', maxsplit=1)
            try:
                self.station = int(station_number)
            except ValueError:
                logging.error("Invalid station number '%s', using platform" % station_number)
                self.station = self.platform
        else:
            self.instrument = args.P
            self.station = self.platform
        logging.debug('instrument=%s, station=%d' % (self.instrument, self.station))

        self.experiment_name = self.get_experiment()
        if self.experiment_name:
            self.last_run_number = self.get_last_run_number()
        else:
            err_msg = 'get_experiment() failed (instrument=\'%s\', station=%d)' % (self.instrument, self.station)
            self.report_error(err_msg)
        logging.debug('__init__(): experiment_name=%s, last_run_number=%d' % (self.experiment_name, self.last_run_number))

        self.ids = set()
        self.handle_request = {
            'selectplatform': self.handle_selectplatform,
            'getstate': self.handle_getstate,
            'storejsonconfig': self.handle_storejsonconfig,
            'getstatus': self.handle_getstatus
        }
        self.handle_fast = {
            'getinstrument': self.handle_getinstrument,
            'getblock': self.handle_getblock
        }
        self.lastTransition = 'reset'
        self.recording = False

        self.collectMachine = Machine(self, ControlDef.states, initial='reset')

        self.collectMachine.add_transition('reset', '*', 'reset',
                                           conditions='condition_reset', after='report_status')
        self.collectMachine.add_transition('rollcall', ['reset', 'unallocated'], 'unallocated',
                                           conditions='condition_rollcall', after='report_status')
        self.collectMachine.add_transition('alloc', 'unallocated', 'allocated',
                                           conditions='condition_alloc', after='report_status')
        self.collectMachine.add_transition('dealloc', 'allocated', 'unallocated',
                                           conditions='condition_dealloc', after='report_status')
        self.collectMachine.add_transition('connect', 'allocated', 'connected',
                                           conditions='condition_connect', after='report_status')
        self.collectMachine.add_transition('disconnect', 'connected', 'allocated',
                                           conditions='condition_disconnect', after='report_status')
        self.collectMachine.add_transition('configure', 'connected', 'configured',
                                           conditions='condition_configure', after='report_status')
        self.collectMachine.add_transition('unconfigure', 'configured', 'connected',
                                           conditions='condition_unconfigure', after='report_status')
        self.collectMachine.add_transition('beginrun', 'configured', 'starting',
                                           conditions='condition_beginrun', after='report_status')
        self.collectMachine.add_transition('endrun', 'starting', 'configured',
                                           conditions='condition_endrun', after='report_status')
        self.collectMachine.add_transition('beginstep', 'starting', 'paused',
                                           conditions='condition_beginstep', after='report_status')
        self.collectMachine.add_transition('endstep', 'paused', 'starting',
                                           conditions='condition_endstep', after='report_status')
        self.collectMachine.add_transition('enable', 'paused', 'running',
                                           after=['after_enable', 'report_status'],
                                           conditions='condition_enable')
        self.collectMachine.add_transition('disable', 'running', 'paused',
                                           conditions='condition_disable', after='report_status')
        # slowupdate is an internal transition
        # do not report status after slowupdate transition
        self.collectMachine.add_transition('slowupdate', 'running', None,
                                           conditions='condition_slowupdate')

        logging.info('Initial state = %s' % self.state)

        if self.slow_update_rate:
            # start slow update thread
            self.set_slow_update_enabled(False)
            self.slow_update_thread.start()

        # start fast reply thread
        self.fast_reply_thread.start()

        # start step done thread
        self.step_done_thread.start()

        # start main loop
        self.run()

        # stop other thread(s)
        self.threads_exit.set()

        self.fast_reply_thread.join()
        if self.slow_update_thread is not None:
            self.slow_update_thread.join()

    #
    # cmstate_levels - return copy of cmstate with only drp/teb/meb entries
    #
    def cmstate_levels(self):
        return {k: self.cmstate[k] for k in self.cmstate.keys() & self.level_keys}

    def service_requests(self):
        # msg['header']['key'] formats:
        #  setstate.STATE
        #  setconfig.CONFIG_ALIAS
        #  TRANSITION
        #  REQUEST
        answer = None
        try:
            msg = self.front_rep.recv_json()
            key = msg['header']['key'].split(".")
            logging.debug("service_requests: key = %s" % key)
            body = msg['body']
            if key[0] == 'setstate':
                # handle_setstate() sends reply internally
                self.phase1Info.update(body)
                self.handle_setstate(key[1])
                answer = None
            elif key[0] == 'setconfig':
                # handle_setconfig() sends reply internally
                self.handle_setconfig(key[1])
                answer = None
            elif key[0] in ControlDef.transitions:
                # is body dict not-empty?
                if body:
                    self.phase1Info[key[1]] = body
                    logging.debug('*** %s %s' % (key[1], phase1Info))
                # send 'ok' reply before calling handle_trigger()
                self.front_rep.send_json(create_msg('ok'))
                # drop slowupdate transition if slowupdate transitions are not enabled,
                # due to race condition between slowupdate and disable
                if key[0] == 'slowupdate' and not self.slow_update_enabled:
                    logging.debug('dropped slowupdate transition in state %s' % self.state)
                    return
                retval = self.handle_trigger(key[0], stateChange=False)
                answer = None
                try:
                    # send error message, if any, to front_pub socket
                    self.report_error(retval['body']['err_info'])
                except KeyError:
                    pass
            else:
                answer = self.handle_request[key[0]](body)
        except KeyError:
            answer = create_msg('error')
        if answer is not None:
            self.front_rep.send_json(answer)

    def service_fast(self):
        # msg['header']['key'] formats:
        #  setrecord.RECORD_FLAG
        #  setbypass.BYPASS_FLAG
        #  REQUEST
        logging.debug('entered service_fast()')
        answer = None
        try:
            msg = self.fast_rep.recv_json()
            key = msg['header']['key'].split(".")
            logging.debug("service_fast: key = %s" % key)
            body = msg['body']

            if key[0] == 'setrecord':
                # handle_setrecord() sends reply internally
                if key[1] == '0':
                    self.handle_setrecord(False)
                else:
                    self.handle_setrecord(True)
                answer = None
            elif key[0] == 'setbypass':
                # handle_setbypass() sends reply internally
                if key[1] == '0':
                    self.handle_setbypass(False)
                else:
                    self.handle_setbypass(True)
                answer = None
            else:
                answer = self.handle_fast[key[0]](body)
                logging.debug(f'service_fast: answer={answer}')

        except KeyError:
            answer = create_msg('error')
        if answer is not None:
            self.fast_rep.send_json(answer)

        return

    #
    # register_file -
    #
    def register_file(self, body):
        if self.experiment_name is None:
            raise ConfigDBError('register_file: experiment_name is None')
            return

        path = body['path']
        logging.info('data file: %s' % path)
        self.front_pub.send_json(fileReport_msg(path))

        # register the file
        # url prefix:     https://pswww.slac.stanford.edu/ws-auth/devlgbk/
        serverURLPrefix = "{0}lgbk/{1}/ws/".format(self.url, self.experiment_name)

        logging.debug('serverURLPrefix = %s' % serverURLPrefix)
        try:
            resp = requests.post(serverURLPrefix + "register_file", json=body,
                                 auth=HTTPBasicAuth(self.user, self.password))
        except Exception as ex:
            raise ConfigDBError('register_file error. HTTP request: %s' % ex)
        else:
            logging.debug("register_file response: %s" % resp.text)
            if resp.status_code == requests.codes.ok:
                if resp.json().get("success", None):
                    logging.debug("register_file success")
                else:
                    raise ConfigDBError("register_file failure")
            else:
                raise ConfigDBError("register_file error: status code %d" % \
                                    resp.status_code)

        return

    #
    # confirm_response -
    #
    def confirm_response(self, socket, wait_time, msg_id, ids, *, progress_txt=None):
        global report_keys
        logging.debug('confirm_response(): ids = %s' % ids)
        msgs = []
        reports = []
        error_flag = False
        begin_time = datetime.now(timezone.utc)
        end_time = begin_time + timedelta(milliseconds=wait_time)
        while len(ids) > 0 and datetime.now(timezone.utc) < end_time and not error_flag:
            if progress_txt is not None:
                self.progressReport(begin_time, end_time, progress_txt=progress_txt)
            for msg in wait_for_answers(socket, 1000, msg_id):

                # exit loop early if an error is received
                if msg['body'] is not None and 'err_info' in msg['body'] and msg['header']['key'] != "warning":
                    logging.debug('confirm_response(): id %s error: %s' %\
                                  (msg['header']['sender_id'], msg['body']['err_info']))
                    ids = [msg['header']['sender_id']]
                    error_flag = True

                if msg['header']['key'] in report_keys:
                    reports.append(msg)
                elif msg['header']['sender_id'] in ids:
                    msgs.append(msg)
                    ids.remove(msg['header']['sender_id'])
                    logging.debug('confirm_response(): removed %s from ids' % msg['header']['sender_id'])
                else:
                    logging.debug('confirm_response(): %s not in ids' % msg['header']['sender_id'])
                if error_flag or len(ids) == 0:
                    break
        for ii in ids:
            logging.debug('id %s did not respond' % ii)
        return ids, msgs, reports

    #
    # process_reports
    #
    def process_reports(self, report_list):
        for msg in report_list:
            try:
                if msg['header']['key'] == 'fileReport':
                    self.register_file(msg['body'])
                elif msg['header']['key'] == 'error':
                    self.report_error(msg['body']['err_info'])
                elif msg['header']['key'] == 'warning':
                    self.report_warning(msg['body']['err_info'])
            except KeyError as ex:
                logging.error('process_reports() KeyError: %s' % ex)

    def service_status(self):
        msg = self.back_pull.recv_json()
        logging.debug('service_status() received msg \'%s\'' % msg)
        self.process_reports([msg])

    def run(self):
        try:
            while True:
                socks = dict(self.poller.poll())
                if self.front_rep in socks and socks[self.front_rep] == zmq.POLLIN:
                    self.service_requests()
                elif self.back_pull in socks and socks[self.back_pull] == zmq.POLLIN:
                    self.service_status()
        except KeyboardInterrupt:
            logging.info('KeyboardInterrupt')

    def handle_trigger(self, key, *, stateChange=True):
        logging.debug('handle_trigger(\'%s\', stateChange=\'%s\') in state \'%s\'' % (key, stateChange, self.state))
        stateBefore = self.state
        trigError = None
        try:
            # this is a call into the "transitions" package which
            # we reuse.  this causes callbacks to happen (e.g.
            # condition_configure()
            self.trigger(key)
        except MachineError as ex:
            logging.debug('MachineError: %s' % ex)
            trigError = str(ex)
        except AttributeError as ex:
            logging.debug('AttributeError: %s' % ex)
            trigError = str(ex)
        else:
            # check for error: trigger failed to change the state
            if stateChange and (self.state == stateBefore):
                trigError = '%s failed to change state' % key

        if trigError is None:
            answer = create_msg(self.state, body=self.cmstate)
        else:
            errMsg = trigError.replace("\"", "")
            answer = create_msg(self.state, body={'err_info': errMsg})

        return answer

    def next_transition(self, oldstate, newstate):
        try:
            retval = next_dict[oldstate][newstate]
        except Exception as ex:
            logging.error('next_transition(\'%s\', \'%s\'): %s' % (oldstate, newstate, ex))
            retval = 'error'

        logging.debug('next_transition(\'%s\', \'%s\') returning \'%s\'' % (oldstate, newstate, retval))
        return retval

    def handle_setstate(self, newstate):
        logging.debug('handle_setstate(\'%s\') in state %s' % (newstate, self.state))
        stateBefore = self.state

        if newstate not in ControlDef.states:
            stateError = 'state \'%s\' not recognized' % newstate
            errMsg = stateError.replace("\"", "")
            logging.error(errMsg)
            answer = create_msg('error', body={'err_info': errMsg})
            # reply 'error'
            self.front_rep.send_json(answer)
        else:
            answer = create_msg('ok')
            # reply 'ok'
            self.front_rep.send_json(answer)
            while self.state != newstate:
                nextT = self.next_transition(self.state, newstate)
                if nextT == 'error':
                    errMsg = 'next_transition() error'
                    logging.error(errMsg)
                    answer = create_msg('error', body={'err_info': errMsg})
                    break
                else:
                    answer = self.handle_trigger(nextT, stateChange=True)
                    if 'err_info' in answer['body']:
                        self.report_error(answer['body']['err_info'])
                        break

        return answer

    def handle_setconfig(self, newconfig):
        logging.debug('handle_setconfig(\'%s\') in state %s' % (newconfig, self.state))

        if self.state == 'running' or self.state == 'paused':
            errMsg = 'cannot set config alias in state \'%s\'' % self.state
            logging.error(errMsg)
            answer = create_msg('error', body={'err_info': errMsg})
            # reply 'error'
            self.front_rep.send_json(answer)
        else:
            if newconfig != self.config_alias:
                self.config_alias = newconfig
                self.report_status()
            answer = create_msg('ok')
            # reply 'ok'
            self.front_rep.send_json(answer)

    def handle_setrecord(self, newrecording):
        logging.debug('handle_setrecord(\'%s\') in state %s' % (newrecording, self.state))

        if self.state == 'running' or self.state == 'paused' or self.state == 'starting':
            errMsg = 'cannot change recording setting in state \'%s\' -- end run first' % self.state
            logging.error(errMsg)
            answer = create_msg('error', body={'err_info': errMsg})
            # reply 'error'
            self.fast_rep.send_json(answer)
        else:
            if newrecording != self.recording:
                self.recording = newrecording
                self.report_status()
            answer = create_msg('ok')
            # reply 'ok'
            self.fast_rep.send_json(answer)

    def handle_setbypass(self, newbypass):
        logging.debug('handle_setbypass(\'%s\') in state %s' % (newbypass, self.state))

        if self.state != 'reset' and self.state != 'unallocated':
            errMsg = 'cannot change bypass_activedet setting in state \'%s\' -- deallocate first' % self.state
            logging.error(errMsg)
            answer = create_msg('error', body={'err_info': errMsg})
            # reply 'error'
            self.fast_rep.send_json(answer)
        else:
            if newbypass != self.bypass_activedet:
                self.bypass_activedet = newbypass
                self.report_status()
            answer = create_msg('ok')
            # reply 'ok'
            self.fast_rep.send_json(answer)

    def status_msg(self):
        if not self.experiment_name:
            expname = 'None'
        else:
            expname = self.experiment_name
        body = {'state': self.state, 'transition': self.lastTransition,
                'platform': self.cmstate_levels(),
                'config_alias': str(self.config_alias), 'recording': self.recording, 'bypass_activedet': self.bypass_activedet,
                'experiment_name': expname, 'run_number': self.run_number, 'last_run_number': self.last_run_number}
        return create_msg('status', body=body)

    def report_status(self):
        logging.debug('status: state=%s transition=%s config_alias=%s recording=%s bypass_activedet=%s' %
                      (self.state, self.lastTransition, self.config_alias, self.recording, self.bypass_activedet))
        self.front_pub.send_json(self.status_msg())

    # check_answers - report and count errors in answers list
    def check_answers(self, answers):
        error_count = 0
        for answer in answers:
            try:
                err_msg = answer['body']['err_info']
                err_sender = answer['header']['sender_id']
                err_alias = self.get_aliases([err_sender]).pop()
                self.report_error('%s: %s' % (err_alias, err_msg))
                error_count = error_count + 1
            except KeyError:
                pass
        return error_count


    def get_phase2_replies(self, transition):
        # get responses from the drp timing systems
        ids = self.filter_active_set(self.ids)
        ids = self.filter_level('drp', ids)
        # make sure all the clients respond to transition before timeout
        missing, answers, reports = self.confirm_response(self.back_pull, self.phase2_timeout, None, ids, progress_txt=transition+' phase 2')
        try:
            self.process_reports(reports)
        except ConfigDBError as ex:
            self.report_error(ex.message)
            return False

        if missing:
            logging.error('%s phase2 failed' % transition)
            for alias in self.get_aliases(missing):
                self.report_error('%s did not respond to %s phase 2' % (alias, transition))
            return False
        return True

    def condition_alloc(self):
        # select procs with active flag set
        ids = self.filter_active_set(self.ids)
        msg = create_msg('alloc', body={'ids': list(ids)})
        self.back_pub.send_multipart([b'all', json.dumps(msg)])

        # make sure all the clients respond to alloc message with their connection info
        retlist, answers, reports = self.confirm_response(self.back_pull, 2000, msg['header']['msg_id'], ids)
        self.process_reports(reports)
        ret = len(retlist)
        if ret:
            for alias in self.get_aliases(retlist):
                self.report_error('%s did not respond to alloc' % alias)
            self.report_error('%d client did not respond to alloc' % ret)
            logging.debug('condition_alloc() returning False')
            return False
        for answer in answers:
            id = answer['header']['sender_id']
            for level, item in answer['body'].items():
                self.cmstate[level][id].update(item)

        active_state = self.filter_active_dict(self.cmstate_levels())
        # give number to drp nodes for the event builder
        if 'drp' in active_state:
            for i, node in enumerate(active_state['drp']):
                self.cmstate['drp'][node]['drp_id'] = i
        else:
            self.report_error('at least one DRP is required')
            logging.debug('condition_alloc() returning False')
            return False

        # assign the readout groups bitmask
        self.groups = get_readout_group_mask(active_state)
        logging.debug('condition_alloc(): groups = 0x%02x' % self.groups)

        # set Disable PV
        if not self.group_run(False):
            logging.error('condition_alloc(): group_run(False) failed')
            return False

        # if you don't want steps, set StepGroups = 0 for each group in partition
        if not self.step_groups_clear(self.groups):
            logging.error('condition_alloc(): step_groups_clear() failed')
            return False

        # create group-dependent PVs
        self.pva.pvListMsgHeader = []
        self.pva.pvListXPM = []
        for g in range(8):
            if self.groups & (1 << g):
                self.pva.pvListMsgHeader.append(self.pva.pv_xpm_base+":PART:"+str(g)+':MsgHeader')
                self.pva.pvListXPM.append(self.pva.pv_xpm_base+":PART:"+str(g)+':Master')
        logging.debug('pvListMsgHeader: %s' % self.pva.pvListMsgHeader)
        logging.debug('pvListXPM: %s' % self.pva.pvListXPM)

        # give number to teb nodes for the event builder
        if 'teb' in active_state:
            for i, node in enumerate(active_state['teb']):
                self.cmstate['teb'][node]['teb_id'] = i
        else:
            self.report_error('at least one TEB is required')
            logging.debug('condition_alloc() returning False')
            return False

        # give number to meb nodes for the event builder
        if 'meb' in active_state:
            for i, node in enumerate(active_state['meb']):
                self.cmstate['meb'][node]['meb_id'] = i
        else:
            self.report_warning('ami NOT supported in absence of MEB')

        logging.debug('cmstate after alloc:\n%s' % self.cmstate)

        # write to the activedet file only if the contents would change
        dst = levels_to_activedet(self.cmstate_levels())
        json_from_file = self.read_json_file(self.activedetfilename)
        if dst == json_from_file:
            logging.debug('condition_alloc(): no change to activedet file %s' % self.activedetfilename)
        else:
            try:
                self.handle_storejsonconfig(oldjson.dumps(dst, sort_keys=True, indent=4))
            except Exception as ex:
                self.report_error('updating activedet file %s' % str(ex))
                return False
            else:
                logging.info('condition_alloc(): updated activedet file %s' % self.activedetfilename)

        self.lastTransition = 'alloc'
        logging.debug('condition_alloc() returning True')
        return True

    def condition_dealloc(self):
        # TODO
        self.lastTransition = 'dealloc'
        logging.debug('condition_dealloc() returning True')
        return True

    def condition_beginrun(self):
        logging.debug('condition_beginrun(): self.recording = %s' % self.recording)

        # run_info
        self.experiment_name = self.get_experiment()
        if not self.experiment_name:
            err_msg = 'condition_beginrun(): get_experiment() failed (instrument=\'%s\', station=%d)' % (self.instrument, self.station)
            self.report_error(err_msg)
            return False

        ok = True
        if self.recording:
            # RECORDING: update runDB
            try:
                self.run_number = self.start_run(self.experiment_name)
            except Exception as ex:
                # ERROR
                self.run_number = 0
                ok = False
                err_msg = "Failed to start a run with recording enabled"
            else:
                self.phase1Info['beginrun'] = {'run_info':{'experiment_name':self.experiment_name, 'run_number':self.run_number}}
                self.runParams.beginrun(self.experiment_name)
        else:
            # NOT RECORDING: by convention, run_number == 0
            self.run_number = 0
            self.phase1Info['beginrun'] = {'run_info':{'experiment_name':self.experiment_name, 'run_number':0}}

        if not ok:
            self.report_error(err_msg)
            return False

        # phase 1
        ok = self.condition_common('beginrun', 6000)
        if not ok:
            logging.error('condition_beginrun(): beginrun phase1 failed')
            return False

        # phase 2
        # ...clear readout
        self.pva.pv_put(self.pva.pvGroupL0Reset, self.groups)
        for pv in self.pva.pvListMsgHeader:
            self.pva.pv_put(pv, ControlDef.transitionId['ClearReadout'])
        self.pva.pv_put(self.pva.pvGroupMsgInsert, self.groups)
        self.pva.pv_put(self.pva.pvGroupMsgInsert, 0)
        time.sleep(1.0)
        for pv in self.pva.pvListMsgHeader:
            self.pva.pv_put(pv, ControlDef.transitionId['BeginRun'])
        self.pva.pv_put(self.pva.pvGroupMsgInsert, self.groups)
        self.pva.pv_put(self.pva.pvGroupMsgInsert, 0)

        self.readoutCumulative = 0

        ok = self.get_phase2_replies('beginrun')
        if not ok:
            return False

        self.lastTransition = 'beginrun'
        return True

    def condition_endrun(self):
        logging.debug('condition_endrun(): self.recording = %s' % self.recording)

        if self.recording and self.experiment_name:
            # update runDB
            self.end_run(self.experiment_name)

        # phase 1
        ok = self.condition_common('endrun', 6000)
        if not ok:
            logging.error('condition_endrun(): endrun phase1 failed')
            return False

        # phase 2
        for pv in self.pva.pvListMsgHeader:
            self.pva.pv_put(pv, ControlDef.transitionId['EndRun'])
        self.pva.pv_put(self.pva.pvGroupMsgInsert, self.groups)
        self.pva.pv_put(self.pva.pvGroupMsgInsert, 0)
        self.step_groups(mask=0)    # default is no scanning

        ok = self.get_phase2_replies('endrun')
        if not ok:
            return False

        self.lastTransition = 'endrun'

        # store last recorded run number
        if self.run_number > 0:
            self.last_run_number = self.run_number
            self.run_number = 0

        return True

    def condition_beginstep(self):
        # phase 1
        ok = self.condition_common('beginstep', 30000)
        if not ok:
            logging.error('condition_beginstep(): beginstep phase1 failed')
            return False

        # phase 2
        for pv in self.pva.pvListMsgHeader:
            self.pva.pv_put(pv, ControlDef.transitionId['BeginStep'])
        self.pva.pv_put(self.pva.pvGroupMsgInsert, self.groups)
        self.pva.pv_put(self.pva.pvGroupMsgInsert, 0)

        ok = self.get_phase2_replies('beginstep')
        if not ok:
            return False

        self.lastTransition = 'beginstep'
        return True

    def condition_endstep(self):
        # phase 1
        ok = self.condition_common('endstep', 6000)
        if not ok:
            logging.error('condition_endstep(): endstep phase1 failed')
            return False

        # phase 2
        for pv in self.pva.pvListMsgHeader:
            self.pva.pv_put(pv, ControlDef.transitionId['EndStep'])
        self.pva.pv_put(self.pva.pvGroupMsgInsert, self.groups)
        self.pva.pv_put(self.pva.pvGroupMsgInsert, 0)

        ok = self.get_phase2_replies('endstep')
        if not ok:
            return False

        self.lastTransition = 'endstep'
        return True

    def condition_slowupdate(self):
        update_ok = True

        # phase 1 not needed
        # phase 2 no replies needed
        for pv in self.pva.pvListMsgHeader:
            if not self.pva.pv_put(pv, ControlDef.transitionId['SlowUpdate']):
                update_ok = False
                break

        if update_ok:
            self.pva.pv_put(self.pva.pvGroupMsgInsert, self.groups)
            self.pva.pv_put(self.pva.pvGroupMsgInsert, 0)
            self.lastTransition = 'slowupdate'

        return update_ok

    def condition_connect(self):
        logging.debug('condition_connect: phase1Info = %s' % self.phase1Info)
        connect_ok = True

        # set XPM PV
        for pv in self.pva.pvListXPM:
            if not self.pva.pv_put(pv, 1):
                self.report_error('connect: failed to put PV \'%s\'' % pv)
                connect_ok = False
                break

        if connect_ok:
            logging.info('master XPM is %d' % self.xpm_master)

            # select procs with active flag set
            ids = self.filter_active_set(self.ids)
            msg = create_msg('connect', body=self.filter_active_dict(self.cmstate_levels()))
            self.back_pub.send_multipart([b'partition', json.dumps(msg)])

            retlist, answers, reports = self.confirm_response(self.back_pull, 15000, msg['header']['msg_id'], ids, progress_txt='connect')
            self.process_reports(reports)
            connect_ok = (self.check_answers(answers) == 0)
            ret = len(retlist)
            if ret:
                for alias in self.get_aliases(retlist):
                    self.report_error('%s did not respond to connect' % alias)
                self.report_error('%d client did not respond to connect' % ret)
                connect_ok = False
            if connect_ok:
                self.lastTransition = 'connect'
        logging.debug('condition_connect() returning %s' % connect_ok)
        return connect_ok

    def condition_disconnect(self):
        # select procs with active flag set
        ids = self.filter_active_set(self.ids)
        msg = create_msg('disconnect')
        self.back_pub.send_multipart([b'partition', json.dumps(msg)])

        retlist, answers, reports = self.confirm_response(self.back_pull, 30000, msg['header']['msg_id'], ids, progress_txt='disconnect')
        self.process_reports(reports)
        disconnect_ok = (self.check_answers(answers) == 0)
        ret = len(retlist)
        if ret:
            for alias in self.get_aliases(retlist):
                self.report_error('%s did not respond to disconnect' % alias)
            self.report_error('%d client did not respond to disconnect' % ret)
            disconnect_ok = False
        if disconnect_ok:
            self.lastTransition = 'disconnect'
        logging.debug('condition_disconnect() returning %s' % disconnect_ok)
        return disconnect_ok

    def handle_getstate(self, body):
        logging.debug('handle_getstate()')
        return create_msg(self.state, body=self.cmstate_levels())

    # returns last transition plus current state
    def handle_getstatus(self, body):
        logging.debug('handle_getstatus()')
        return self.status_msg()

    # Update the active detector file.
    # May throw an exception.
    def handle_storejsonconfig(self, body):
        logging.debug('handle_storejsonconfig(): body = %s' % body)
        if self.activedetfilename != '/dev/null':
            with open(self.activedetfilename, 'w') as f:
                print('%s' % body, file=f)

        return {}

    def handle_getinstrument(self, body):
        logging.debug('handle_getinstrument()')
        body = {'instrument': self.instrument, 'station': self.station}
        return create_msg('instrument', body=body)

    def handle_getblock(self, body):
        if body is None or type(body) != type({}):
            msg = 'getblock requires dict'
            self.report_error(msg)
            return error_msg(msg)

        logging.debug(f'handle_getblock: body={body}')

        try:
            detname       = body["detname"]
            dettype       = body["dettype"]
            serial_number = body["serial_number"]
            namesid       = body["namesid"]

            nameinfo      = dc.nameinfo(detname,dettype,serial_number,namesid)
            alg           = dc.alg(body["alg_name"], body["alg_version"])
            self.cydgram.addDet(nameinfo, alg, body["motors"])

            # create dgram
            add_names       = body["add_names"]
            add_shapes_data = body["add_shapes_data"]
            timestamp       = body["timestamp"]
            transitionid    = body["transitionid"]
        except Exception as ex:
            logging.error(f'handle_getblock Exception: {ex}')

        xtc_bytes = self.cydgram.getSelect(timestamp, transitionid, add_names=add_names, add_shapes_data=add_shapes_data)
        logging.debug('handle_getblock: transitionid %d dgram is %d bytes (with header)' % (transitionid, len(xtc_bytes)))

        # remove first 12 bytes (dgram header), and keep next 12 bytes (xtc header)
        reply = xtc_bytes[12:]
        return create_msg('block', body=reply.hex())

    def handle_selectplatform(self, body):
        logging.debug('handle_selectplatform()')
        if self.state != 'unallocated':
            msg = 'selectPlatform only permitted in unallocated state'
            self.report_error(msg)
            return error_msg(msg)

        try:
            for level, val1 in body.items():
                for key2, val2 in val1.items():
                    if level == 'control' and body[level][key2]['active'] == 0:
                        self.report_warning('ignoring attempt to clear the control level active flag')
                        body[level][key2]['active'] = 1
                    self.cmstate[level][int(key2)]['active'] = body[level][key2]['active']
                    if level == 'drp':
                        # drp readout group
                        if self.cmstate[level][int(key2)]['active'] == 1:
                            self.cmstate[level][int(key2)]['det_info']['readout'] = body[level][key2]['det_info']['readout']
                        else:
                            self.cmstate[level][int(key2)]['det_info']['readout'] = self.platform

        except Exception as ex:
            msg = 'handle_selectplatform(): %s' % ex
            logging.error(msg)
            return error_msg(msg)

        return create_msg('ok')

    def on_enter_reset(self):
        self.cmstate.clear()
        self.ids.clear()
        return

    def subtract_clients(self, missing_set):
        if missing_set:
            for level, item in self.cmstate_levels().items():
                for xid in item.keys():
                    try:
                        alias = item[xid]['proc_info']['alias']
                    except KeyError as ex:
                        logging.error('KeyError: %s' % ex)
                    else:
                        missing_set -= set(['%s/%s' % (level, alias)])
        return

    def read_json_file(self, filename):
        json_data = {}

        if not os.path.isfile(filename):
            if filename == '/dev/null':
                self.report_warning('/dev/null is not a proper active detectors file')
            else:
                self.report_error('active detectors file %s not found' % filename)
            return {}

        if os.path.getsize(filename) == 0:
            self.report_error('active detectors file %s is empty' % filename)
            return {}

        try:
            with open(filename) as fd:
                json_data = oldjson.load(fd)
        except Exception as ex:
            self.report_error("read_json_file(%s): %s" % (filename, ex))
            json_data = {}
        else:
            logging.info('read_json_file(%s): json_data:\n%s' % (filename, json_data))

        return json_data

    def get_required_set(self, d):
        retval = set()
        for level, item1 in d["activedet"].items():
            for alias, item2 in item1.items():
                if item2["active"]:
                    retval.add(level + "/" + alias)
        return retval

    def progressReport(self, begin_time, end_time, *, progress_txt):
        elapsed = (datetime.now(timezone.utc) - begin_time).total_seconds()
        if elapsed >= 1.0:
            total   = (end_time - begin_time).total_seconds()
            self.front_pub.send_json(progress_msg(progress_txt, elapsed, total))
        return

    def condition_rollcall(self):
        global report_keys
        retval = False
        required_set = set()

        if not self.bypass_activedet and not os.path.isfile(self.activedetfilename):
            self.report_error('Missing active detectors file %s' % self.activedetfilename)
            logging.warning("active detectors file disabled. Default settings will be used.")
            # active detectors file bypassed
            self.bypass_activedet = True

        if not self.bypass_activedet:
            # determine which clients are required by reading the active detectors file
            json_data = self.read_json_file(self.activedetfilename)
            if len(json_data) > 0:
                if "activedet" in json_data.keys():
                    required_set = self.get_required_set(json_data)
                else:
                    self.report_error('Missing "activedet" key in active detectors file %s' % self.activedetfilename)
            if not required_set:
                self.report_error('Failed to read configuration from active detectors file %s' % self.activedetfilename)

        logging.debug('rollcall: bypass_activedet = %s' % self.bypass_activedet)
        missing_set = required_set.copy()
        newfound_set = set()
        self.cmstate.clear()
        self.ids.clear()
        msg = create_msg('rollcall')
        begin_time = datetime.now(timezone.utc)
        end_time = begin_time + timedelta(seconds=self.rollcall_timeout)
        while datetime.now(timezone.utc) < end_time:
            self.back_pub.send_multipart([b'all', json.dumps(msg)])
            for answer in wait_for_answers(self.back_pull, 1000, msg['header']['msg_id']):
                if answer['header']['key'] in report_keys:
                    self.process_reports([answer])
                    continue
                for level, item in answer['body'].items():
                    alias = item['proc_info']['alias']
                    responder = level + '/' + alias
                    if not self.bypass_activedet:
                        if responder not in required_set:
                            if responder not in newfound_set:
                                newfound_set.add(responder)
                            elif responder not in missing_set:
                                # ignore duplicate response
                                continue
                    if level not in self.cmstate:
                        self.cmstate[level] = {}
                    id = answer['header']['sender_id']
                    self.cmstate[level][id] = item
                    if level == 'drp' or level == 'meb':
                        self.cmstate[level][id]['hidden'] = 0
                    else:
                        self.cmstate[level][id]['hidden'] = 1
                    logging.debug('rollcall: responder (%s) in newfound_set = %s' % (responder, responder in newfound_set))
                    if self.bypass_activedet:
                        # active detectors file disabled: default to active=1
                        self.cmstate[level][id]['active'] = 1
                        if level == 'drp':
                            self.cmstate[level][id]['det_info'] = {}
                            self.cmstate[level][id]['det_info']['readout'] = self.platform
                    elif responder in newfound_set:
                        # new detector or meb + active detectors file enabled: default to active=0
                        if level == 'drp' or level == 'meb':
                            self.cmstate[level][id]['active'] = 0
                            self.report_warning('rollcall: %s NOT selected for data collection' % responder)
                            if level == 'drp':
                                self.cmstate[level][id]['det_info'] = {}
                                self.cmstate[level][id]['det_info']['readout'] = self.platform
                        else:
                            # neither detector nor meb: default to active=1
                            self.cmstate[level][id]['active'] = 1
                    else:
                        # copy values from active detectors file
                        self.cmstate[level][id]['active'] = json_data['activedet'][level][alias]['active']
                        if level == 'drp':
                            self.cmstate[level][id]['det_info'] = json_data['activedet'][level][alias]['det_info'].copy()
                            group = json_data['activedet'][level][alias]['det_info']['readout']
                            logging.info('rollcall: %s selected for data collection (readout group %d)' % (responder, group))
                        else:
                            logging.info('rollcall: %s selected for data collection' % responder)
                    self.ids.add(id)
            self.subtract_clients(missing_set)
            if not missing_set:
                break
            self.progressReport(begin_time, end_time, progress_txt='rollcall')

        for dup in self.check_for_dups():
            self.report_error('duplicate alias responded to rollcall: %s' % dup)

        if missing_set:
            for client in missing_set:
                self.report_error(client + ' did not respond to rollcall')
            # Despite rollcall transition errors, allow state machine to advance.
            retval = True
            self.lastTransition = 'rollcall'
        else:
            retval = True
            self.lastTransition = 'rollcall'

        # add control info
        if not 'control' in self.cmstate:
            self.cmstate['control'] = {}
            self.cmstate['control'][0] = {}
            self.cmstate['control'][0]['active'] = 1
            self.cmstate['control'][0]['hidden'] = 1
            self.cmstate['control'][0]['control_info'] = {}
            self.cmstate['control'][0]['proc_info'] = {}
            self.cmstate['control'][0]['control_info']['xpm_master'] = self.xpm_master
            self.cmstate['control'][0]['control_info']['pv_base'] = self.pv_base
            self.cmstate['control'][0]['control_info']['cfg_dbase'] = self.cfg_dbase
            self.cmstate['control'][0]['control_info']['instrument'] = self.instrument
            self.cmstate['control'][0]['control_info']['slow_update_rate'] = self.slow_update_rate
            self.cmstate['control'][0]['proc_info']['alias'] = self.alias
            self.cmstate['control'][0]['proc_info']['host'] = socket.gethostname()
            self.cmstate['control'][0]['proc_info']['pid'] = os.getpid()

        logging.debug('cmstate after rollcall:\n%s' % self.cmstate)
        logging.debug('condition_rollcall() returning %s' % retval)
        return retval

    # check_for_dups - check for duplicate aliases
    def check_for_dups(self):
        aliases = set()
        dups = set()
        for level, item in self.cmstate_levels().items():
            for xid in item:
                alias = self.cmstate[level][xid]['proc_info']['alias']
                if alias in aliases:
                    dups.add(level + '/' + alias)
                else:
                    aliases.add(alias)
        if len(dups) > 0:
            logging.debug('duplicate aliases: %s' % dups)
        return dups

    # filter_active_set - return subset of ids which have 'active' flag set
    def filter_active_set(self, ids):
        matches = set()
        for level, item in self.cmstate_levels().items():
            for xid in item:
                if item[xid]['active'] == 1:
                    matches.add(xid)
        return matches.intersection(ids)

    # filter_active_dict - return subset of dict that has 'active' flag set
    def filter_active_dict(self, oldstate):
        newstate = dict()
        for level, item in oldstate.items():
            for xid in item:
                if item[xid]['active'] == 1:
                    if level not in newstate:
                        newstate[level] = dict()
                    newstate[level][xid] = copy.copy(oldstate[level][xid])
        return newstate

    # filter_level - return subset of ids for which 'level' starts with prefix
    def filter_level(self, prefix, ids):
        matches = set()
        for level, item in self.cmstate_levels().items():
            if level.startswith(prefix):
                matches.update(set(item.keys()))
        return matches.intersection(ids)

    def get_aliases(self, id_list):
        alias_list = []
        for level, item in self.cmstate_levels().items():
            for xid in item.keys():
                if xid in id_list and 'proc_info' in item[xid]:
                    try:
                        alias_list.append(item[xid]['proc_info']['alias'])
                    except KeyError:
                        alias_list.append('%s/%s/%s' %
                            (level,
                             item[xid]['proc_info']['pid'],
                             item[xid]['proc_info']['host']))
        return alias_list

    def report_error(self, msg):
        logging.error(msg)
        self.front_pub.send_json(error_msg(msg))
        return

    def report_warning(self, msg):
        logging.warning(msg)
        self.front_pub.send_json(warning_msg(msg))
        return

    def start_run(self, experiment_name):
        run_num = 0
        ok = False
        serverURLPrefix = "{0}run_control/{1}/ws/".format(self.url + "/" if not self.url.endswith("/") else self.url, experiment_name)
        logging.debug('serverURLPrefix = %s' % serverURLPrefix)
        try:
            resp = requests.post(serverURLPrefix + "start_run", auth=HTTPBasicAuth(self.user, self.password))
        except Exception as ex:
            logging.error("start_run (user=%s) exception: %s" % (self.user, ex))
        else:
            logging.debug("start_run response: %s" % resp.text)
            if resp.status_code == requests.codes.ok:
                if resp.json().get("success", None):
                    logging.debug("start_run success")
                    run_num = resp.json().get("value", {}).get("num", None)
                    ok = True
            else:
                self.report_error("start_run (user=%s) error: status code %d" % (self.user, resp.status_code))

        if not ok:
            raise Exception("start_run error")

        logging.info("start_run: run number = %s" % run_num)
        return run_num

    def add_run_params(self, experiment_name, params):
        param_count = len(params)
        if param_count > 0:
            ok = False
            err_msg = "add_run_params error"
            serverURLPrefix = "{0}run_control/{1}/ws/".format(self.url + "/" if not self.url.endswith("/") else self.url, experiment_name)
            logging.debug('serverURLPrefix = %s' % serverURLPrefix)
            try:
                resp = requests.post(serverURLPrefix + "add_run_params", json=params, auth=HTTPBasicAuth(self.user, self.password))
            except Exception as ex:
                err_msg = "add_run_params error (user=%s): %s" % (self.user, ex)
            else:
                logging.debug("add_run_params response: %s" % resp.text)
                if resp.status_code == requests.codes.ok:
                    if resp.json().get("success", None):
                        logging.debug("add_run_params success")
                        ok = True
                else:
                    err_msg = "add_run_params error (user=%s): status code %d" % (self.user, resp.status_code)
            if not ok:
                param_count = 0
                self.report_error(err_msg)
        return param_count

    def add_update_run_param_descriptions(self, experiment_name, param_descs):
        param_desc_count = len(param_descs)
        if param_desc_count > 0:
            ok = False
            err_msg = "add_update_run_param_descriptions error"
            serverURLPrefix = "{0}run_control/{1}/ws/".format(self.url + "/" if not self.url.endswith("/") else self.url, experiment_name)
            logging.debug('serverURLPrefix = %s' % serverURLPrefix)
            try:
                resp = requests.post(serverURLPrefix + "add_update_run_param_descriptions", json=param_descs, auth=HTTPBasicAuth(self.user, self.password))
            except Exception as ex:
                err_msg = "add_update_run_param_descriptions error (user=%s): %s" % (self.user, ex)
            else:
                logging.debug("add_update_run_param_descriptions response: %s" % resp.text)
                if resp.status_code == requests.codes.ok:
                    if resp.json().get("success", None):
                        logging.debug("add_update_run_param_descriptions success")
                        ok = True
                else:
                    err_msg = "add_update_run_param_descriptions error (user=%s): status code %d" % (self.user, resp.status_code)
            if not ok:
                param_desc_count = 0
                self.report_error(err_msg)
        return param_desc_count

    def end_run(self, experiment_name):
        run_num = 0
        ok = False
        err_msg = "end_run error"
        serverURLPrefix = "{0}run_control/{1}/ws/".format(self.url + "/" if not self.url.endswith("/") else self.url, experiment_name)
        logging.debug('serverURLPrefix = %s' % serverURLPrefix)
        try:
            resp = requests.post(serverURLPrefix + "end_run", auth=HTTPBasicAuth(self.user, self.password))
        except Exception as ex:
            err_msg = "end_run error (user=%s): %s" % (self.user, ex)
        else:
            logging.debug("Response: %s" % resp.text)
            if resp.status_code == requests.codes.ok:
                if resp.json().get("success", None):
                    logging.debug("end_run success")
                    ok = True
            else:
                err_msg = "end_run error (user=%s): status code %d" % (self.user, resp.status_code)

        if not ok:
            self.report_error(err_msg)
        return

    def get_last_run_number(self):
        logging.debug('get_last_run_number()')
        last_run_number = 0

        # authentication is not required, adjust url accordingly
        uurl = self.url.replace('ws-auth', 'ws').replace('ws-kerb', 'ws')

        try:
            resp = requests.get((uurl + "/" if not uurl.endswith("/") else uurl) + \
                                "lgbk/" + self.experiment_name + "/ws/current_run",
                                timeout=10)
        except requests.exceptions.RequestException as ex:
            logging.error("get_last_run_number(): request exception: %s" % ex)
        except Exception as ex:
            logging.error("get_last_run_number(): exception: %s" % ex)
        else:
            logging.debug("current_run request response: %s" % resp.text)
            if resp.status_code == requests.codes.ok:
                logging.debug("current_run request response headers: %s" % resp.headers)
                if 'application/json' in resp.headers['Content-Type']:
                    try:
                        json_response = resp.json()
                    except json.decoder.JSONDecodeError:
                        logging.error("Error: failed to decode JSON")
                    else:
                        if json_response is None or json_response.get("value", {}) is None:
                            logging.debug("get_last_run_number(): JSON response is None")
                        else:
                            try:
                                last_run_number = json_response.get("value", {}).get("num", 0)
                            except Exception as ex:
                                logging.error("get_last_run_number(): failed to get num: %s" % ex)
                else:
                    logging.error("Error: failed to receive JSON")
            else:
                logging.error("Error: status code %d" % resp.status_code)

        # last run number, or 0
        return last_run_number

    def get_experiment(self):
        logging.debug('get_experiment()')
        experiment_name = None
        instrument = self.instrument

        # authentication is not required, adjust url accordingly
        uurl = self.url.replace('ws-auth', 'ws').replace('ws-kerb', 'ws')

        try:
            resp = requests.get((uurl + "/" if not uurl.endswith("/") else uurl) + "/lgbk/ws/activeexperiment_for_instrument_station",
                                params={"instrument_name": instrument, "station": self.station}, timeout=10)
        except requests.exceptions.RequestException as ex:
            logging.error("get_experiment(): request exception: %s" % ex)
        else:
            logging.debug("request response: %s" % resp.text)
            if resp.status_code == requests.codes.ok:
                logging.debug("headers: %s" % resp.headers)
                if 'application/json' in resp.headers['Content-Type']:
                    try:
                        experiment_name = resp.json().get("value", {}).get("name", None)
                    except json.decoder.JSONDecodeError:
                        logging.error("Error: failed to decode JSON")
                else:
                    logging.error("Error: failed to receive JSON")
            else:
                logging.error("Error: status code %d" % resp.status_code)

        # result of request, or None
        return experiment_name

    def condition_common(self, transition, timeout, body=None):
        if body is None:
            body = {}
        retval = True
        # select procs with active flag set
        ids = self.filter_active_set(self.ids)
        # include phase1 info in the msg, if it exists
        if transition in self.phase1Info.keys():
            body['phase1Info'] = self.phase1Info[transition]
            logging.debug('condition_common(%s): body = %s' % (transition, body))
        msg = create_msg(transition, body=body)
        self.back_pub.send_multipart([b'partition', json.dumps(msg)])
        # now that the message has been sent, delete the phase1
        # info so we don't send stale information next time.
        self.phase1Info.pop(transition,None)

        # only drp/teb/meb groups (aka levels) respond to configure and above
        ids = self.filter_level('drp', ids) | self.filter_level('teb', ids) | self.filter_level('meb',ids)

        if len(ids) == 0:
            logging.debug('condition_common() empty set of ids')
            return True

        # make sure all the clients respond to transition before timeout
        retlist, answers, reports = self.confirm_response(self.back_pull, timeout, msg['header']['msg_id'], ids, progress_txt=transition)
        self.process_reports(reports)
        answers_ok = (self.check_answers(answers) == 0)
        ret = len(retlist)
        if ret:
            # Error
            retval = False
            for alias in self.get_aliases(retlist):
                self.report_error('%s did not respond to %s' % (alias, transition))
            self.report_error('%d client did not respond to %s' % (ret, transition))
        elif not answers_ok:
            # Error
            retval = False
        return retval

    def condition_configure(self):
        logging.debug('condition_configure: phase1Info = %s' % self.phase1Info)
        # phase 1
        ok = self.condition_common('configure', 45000,
                                   body={'config_alias': self.config_alias, 'trigger_config': self.trigger_config})
        if not ok:
            logging.error('condition_configure(): configure phase1 failed')
            return False

        self.runParams.configure()

        # phase 2
        # ...clear readout
        self.pva.pv_put(self.pva.pvGroupL0Reset, self.groups)
        for pv in self.pva.pvListMsgHeader:
            self.pva.pv_put(pv, ControlDef.transitionId['ClearReadout'])
        self.pva.pv_put(self.pva.pvGroupMsgInsert, self.groups)
        self.pva.pv_put(self.pva.pvGroupMsgInsert, 0)
        time.sleep(1.0)
        # ...configure
        for pv in self.pva.pvListMsgHeader:
            self.pva.pv_put(pv, ControlDef.transitionId['Configure'])
        self.pva.pv_put(self.pva.pvGroupMsgInsert, self.groups)
        self.pva.pv_put(self.pva.pvGroupMsgInsert, 0)
        self.step_groups(mask=0)    # default is no scanning

        self.readoutCumulative = 0

        ok = self.get_phase2_replies('configure')
        if not ok:
            return False

        logging.debug('condition_configure() returning %s' % ok)

        self.lastTransition = 'configure'
        return True

    def condition_unconfigure(self):
        self.phase1Info = {}    # clear phase1Info

        self.runParams.unconfigure()
        # phase 1
        ok = self.condition_common('unconfigure', 6000)
        if not ok:
            logging.error('condition_unconfigure(): unconfigure phase1 failed')
            return False

        # phase 2
        for pv in self.pva.pvListMsgHeader:
            self.pva.pv_put(pv, ControlDef.transitionId['Unconfigure'])
        self.pva.pv_put(self.pva.pvGroupMsgInsert, self.groups)
        self.pva.pv_put(self.pva.pvGroupMsgInsert, 0)

        ok = self.get_phase2_replies('unconfigure')
        if not ok:
            return False

        logging.debug('condition_unconfigure() returning %s' % ok)

        self.lastTransition = 'unconfigure'
        return True

    def group_run(self, enable):
        if enable:
            rv = self.pva.pv_put(self.pva.pvGroupL0Enable, self.groups)
        else:
            rv = self.pva.pv_put(self.pva.pvGroupL0Disable, self.groups)
        return rv

    # step_groups_clear - clear all StepGroups PVs included in mask
    # Returns False on error
    def step_groups_clear(self, groups):
        logging.debug("step_groups_clear()")
        retval = True
        for g in range(8):
            if groups & (1 << g):
                pv = self.pva.pv_xpm_base + f':PART:{g}:StepGroups'
                logging.debug(f'step_groups_clear(): clearing {pv}')
                if not self.pva.pv_put(pv, 0):
                    logging.error(f'step_groups_clear(): clearing {pv} failed')
                    retval = False

        return retval

    # step_groups -
    # Returns False on error
    def step_groups(self, *, mask):
        logging.debug("step_groups(mask=%d)" % mask)
        return self.pva.pv_put(self.pva.pvStepGroups, mask)

    # set slow_update_enabled to True or False
    def set_slow_update_enabled(self, enabled):
        self.slow_update_enabled = enabled
        if enabled:
            logging.info('slowupdate transitions ENABLED')
        else:
            logging.info('slowupdate transitions DISABLED')

    def after_enable(self):
        if self.slow_update_rate:
            # enable slowupdate transitions
            self.set_slow_update_enabled(True)

    def condition_enable(self):
        # readout_count and group_mask are optional
        try:
            readout_count = self.phase1Info['enable']['readout_count']
            group_mask = self.phase1Info['enable']['group_mask']
        except KeyError:
            readout_count = 0
            group_mask = 1 << self.platform
        logging.debug(f'condition_enable(): readout_count={readout_count} group_mask={group_mask}')

        # phase 1
        ok = self.condition_common('enable', 6000)
        if not ok:
            logging.error('condition_enable(): enable phase1 failed')
            return False

        # phase 2
        if (readout_count > 0):
            # set EPICS PVs.
            # StepEnd is a cumulative count.
            self.readoutCumulative += readout_count
            self.pva.pv_put(self.pva.pvStepEnd, self.readoutCumulative)
            self.pva.step_groups(mask=group_mask)
            self.pva.pv_put(self.pva.pvStepDone, 0)
        else:
            self.step_groups(mask=0)    # default is no scanning

        for pv in self.pva.pvListMsgHeader:
            self.pva.pv_put(pv, ControlDef.transitionId['Enable'])
        self.pva.pv_put(self.pva.pvGroupMsgInsert, self.groups)
        self.pva.pv_put(self.pva.pvGroupMsgInsert, 0)

        ok = self.get_phase2_replies('enable')
        if not ok:
            return False

        # order matters: set Enable PV after others transition
        if not self.group_run(True):
            logging.error('condition_enable(): group_run(True) failed')
            return False

        self.lastTransition = 'enable'
        return True


    def condition_disable(self):

        # order matters: set Disable PV before others transition
        if not self.group_run(False):
            logging.error('condition_disable(): group_run(False) failed')
            return False

        # disable slow updates early in the disable transition
        # but after setting Disable PV has succeeded
        if self.slow_update_rate:
            self.set_slow_update_enabled(False)

        # phase 1
        ok = self.condition_common('disable', 6000)
        if not ok:
            logging.error('condition_disable(): disable phase1 failed')
            logging.warning('condition_disable(): L0s are disabled')
            if self.slow_update_rate and not self.slow_update_enabled:
                logging.warning('condition_disable(): slowupdate transitions are disabled')
            return False

        # phase 2
        for pv in self.pva.pvListMsgHeader:
            self.pva.pv_put(pv, ControlDef.transitionId['Disable'])
        self.pva.pv_put(self.pva.pvGroupMsgInsert, self.groups)
        self.pva.pv_put(self.pva.pvGroupMsgInsert, 0)

        ok = self.get_phase2_replies('disable')
        if not ok:
            return False

        self.lastTransition = 'disable'
        return True


    def condition_reset(self):
        self.phase1Info = {}    # clear phase1Info

        # disable triggers
        if self.state == 'running':
            if not self.group_run(False):
                logging.error('condition_reset(): group_run(False) failed')

        # disable slowupdate timer
        self.set_slow_update_enabled(False)

        msg = create_msg('reset')
        self.back_pub.send_multipart([b'all', json.dumps(msg)])
        self.lastTransition = 'reset'
        return True

    def slow_update_func(self):
        logging.debug('slowupdate thread starting up')

        # zmq sockets are not thread-safe
        # so create a zmq socket for the slowupdate thread
        slow_front_req = self.context.socket(zmq.REQ)
        slow_front_req.connect('tcp://localhost:%d' % front_rep_port(self.platform))
        msg = create_msg('slowupdate')

        while not self.threads_exit.wait(1.0 / self.slow_update_rate):
            if self.slow_update_enabled:
                slow_front_req.send_json(msg)
                answer = slow_front_req.recv_multipart()

        logging.debug('slowupdate thread shutting down')

    def fast_reply_func(self):
        logging.debug('fastreply thread starting up')

        # zmq sockets are not thread-safe
        # so create a zmq socket for the fastreply thread
        fast_reply_rep = self.context.socket(zmq.REP)

        while not self.threads_exit.wait(1.0 / self.fast_reply_rate):
            socks = dict(self.fast_poller.poll(500))        # timeout (ms)
            if self.fast_rep in socks and socks[self.fast_rep] == zmq.POLLIN:
                self.service_fast()
            if self.threads_exit.is_set():
                break

        logging.debug('fastreply thread shutting down')

    def step_done_func(self):
        logging.debug('stepdone thread starting up')

        # zmq sockets are not thread-safe
        # so create a zmq socket for the stepdone thread
        self.context = zmq.Context(1)
        self.step_pub = self.context.socket(zmq.PUB)
        self.step_pub.bind('tcp://*:%d' % step_pub_port(self.platform))

        # define nested function for monitoring the StepDone PV
        def callback(done):
            doneFlag = int(done)
            if doneFlag:
                if self.state != 'running':
                    logging.debug(f'StepDone PV={doneFlag} in state {self.state} (ignore)')
                elif doneFlag:
                    logging.debug(f'StepDone PV={doneFlag} in state {self.state} (set step_done event)')
                    self.step_done.set()

        # start monitoring the StepDone PV
        sub = self.pva.monitor_StepDone(callback=callback)

        while not self.threads_exit.is_set():
            if self.step_done.wait(0.5):
                self.step_done.clear()
                # stepDone event received
                logging.debug('stepDone event received')
                # publish the stepDone event with zmq
                self.step_pub.send_json(step_msg(1))

        # stop monitoring the StepDone PV
        sub.close()

        logging.debug('stepdone thread shutting down')


def main():
    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-x', metavar='XPM', type=int, required=True, help='master XPM')
    parser.add_argument('-P', metavar='INSTRUMENT', required=True, help='instrument_name[:station_number]')
    parser.add_argument('-d', metavar='CFGDATABASE', default='https://pswww.slac.stanford.edu/ws/devconfigdb/ws/configDB', help='configuration database connection')
    parser.add_argument('-B', metavar='PVBASE', required=True, help='PV base')
    parser.add_argument('-u', metavar='ALIAS', required=True, help='unique ID')
    parser.add_argument('-C', metavar='CONFIG_ALIAS', required=True, help='default configuration type (e.g. ''BEAM'')')
    parser.add_argument('-t', metavar='TRIGGER_CONFIG', default='tmoteb', help='trigger configuration name')
    parser.add_argument('-S', metavar='SLOW_UPDATE_RATE', type=int, default=0, choices=(0, 1, 5, 10), help='slow update rate (Hz, default 0)')
    parser.add_argument('-T', type=int, metavar='P2_TIMEOUT', default=7500, help='phase 2 timeout msec (default 7500)')
    parser.add_argument('--rollcall_timeout', type=int, default=30, help='rollcall timeout sec (default 30)')
    parser.add_argument('-v', action='store_true', help='be verbose')
    parser.add_argument('-V', metavar='LOGBOOK_FILE', default='/dev/null', help='run parameters file')
    parser.add_argument("--user", default="tstopr", help='HTTP authentication user')
    parser.add_argument("--password", default="pcds", help='HTTP authentication password')
    defaultURL = "https://pswww.slac.stanford.edu/ws-auth/devlgbk/"
    parser.add_argument("--url", help="run database URL prefix. Defaults to " + defaultURL, default=defaultURL)
    defaultActiveDetFile = "~/.psdaq/p<platform>.activedet.json"
    parser.add_argument('-r', metavar='ACTIVEDETFILE', help="active detectors file. Defaults to " + defaultActiveDetFile)
    args = parser.parse_args()

    # configure logging handlers
    if args.v:
        level=logging.DEBUG
    else:
        level=logging.INFO
    logger = SysLog(instrument=args.P, level=level)
    logging.info('logging initialized')

    try:
        manager = CollectionManager(args)
    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt')

if __name__ == '__main__':
    main()
