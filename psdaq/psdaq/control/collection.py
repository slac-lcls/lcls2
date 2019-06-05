import os
import time
import copy
import socket
from datetime import datetime, timezone
import zmq
import zmq.utils.jsonapi as json
from transitions import Machine, MachineError, State
import argparse
import logging
from p4p.client.thread import Context

PORT_BASE = 29980
POSIX_TIME_AT_EPICS_EPOCH = 631152000

class DaqControl:
    'Base class for controlling data acquisition'

    # transitionId is a subset of the TransitionId.hh enum
    transitionId = {
        'ClearReadout'      : 0,
        'Reset'             : 1,
        'Configure'         : 2,
        'Unconfigure'       : 3,
        'Enable'            : 4,
        'Disable'           : 5,
        'ConfigUpdate'      : 6,
        'BeginRecord'       : 7,
        'EndRecord'         : 8,
        'L1Accept'          : 12,
    }

    transitions = ['plat', 'alloc', 'dealloc',
                   'connect', 'disconnect',
                   'configure', 'unconfigure',
                   'enable', 'disable',
                   'configupdate', 'reset']

    states = [
        'reset',
        'unallocated',
        'allocated',
        'connected',
        'paused',
        'running'
    ]

    default_active = 1
    # default readout group is self.platform

    def __init__(self, *, host, platform, timeout):
        self.host = host
        self.platform = platform
        self.timeout = timeout

        # initialize zmq socket
        self.context = zmq.Context(1)
        self.front_sub = self.context.socket(zmq.SUB)
        self.front_sub.connect('tcp://%s:%d' % (host, front_pub_port(platform)))
        self.front_sub.setsockopt(zmq.SUBSCRIBE, b'')
        self.front_req = self.context.socket(zmq.REQ)
        self.front_req.linger = 0
        self.front_req.RCVTIMEO = timeout # in milliseconds
        self.front_req.connect('tcp://%s:%d' % (host, front_rep_port(platform)))

    #
    # DaqControl.getState - get current state
    #
    def getState(self):
        retval = 'error'
        try:
            msg = create_msg('getstate')
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            print('getState() Exception: %s' % ex)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        else:
            try:
                retval = reply['header']['key']
            except KeyError:
                pass

        return retval

    #
    # DaqControl.getPlatform - get platform
    #
    def getPlatform(self):
        retval = {}
        try:
            msg = create_msg('getstate')
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            print('getPlatform() Exception: %s' % ex)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        else:
            try:
                retval = reply['body']
            except KeyError:
                pass

        return retval

    #
    # DaqControl.selectPlatform - select platform
    #
    def selectPlatform(self, body):
        retval = {}
        try:
            msg = create_msg('selectplatform', body=body)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            print('selectPlatform(): %s' % ex)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        else:
            try:
                retval = reply['body']
            except KeyError:
                pass

        return retval

    #
    # DaqControl.getInstrument - get instrument name
    #
    def getInstrument(self):
        r1 = None
        try:
            msg = create_msg('getinstrument')
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            print('getInstrument() Exception: %s' % ex)
        else:
            try:
                r1 = reply['body']['instrument']
            except Exception as ex:
                print('getInstrument() Exception: %s' % ex)

        return r1

    #
    # DaqControl.getStatus - get status
    #
    def getStatus(self):
        r1 = r2 = r3 = 'error'
        try:
            msg = create_msg('getstatus')
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            print('getStatus() Exception: %s' % ex)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        else:
            try:
                r1 = reply['body']['transition']
                r2 = reply['body']['state']
                r3 = reply['body']['config_alias']
            except KeyError:
                pass

        return (r1, r2, r3)

    #
    # DaqControl.monitorStatus - monitor the status
    #
    def monitorStatus(self):

        # process messages
        while True:
            try:
                msg = self.front_sub.recv_json()

                if msg['header']['key'] == 'status':
                    # return transition, state, config_alias
                    return msg['body']['transition'], msg['body']['state'], msg['body']['config_alias']

                elif msg['header']['key'] == 'error':
                    # return 'error', error message, 'error'
                    return 'error', msg['body']['err_info'], 'error'

            except KeyboardInterrupt:
                break

            except KeyError as ex:
                logging.error('KeyError: %s' % ex)
                break

        return None, None, None

    #
    # DaqControl.setState - change the state
    #
    def setState(self, state, *, config_alias='BEAM'):
        errorMessage = None
        try:
            msg = create_msg('setstate.' + state + '.' + config_alias)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            errorMessage = 'setState() Exception: %s' % ex
        else:
            try:
                errorMessage = reply['body']['err_info']
            except KeyError:
                pass

        return errorMessage

    #
    # DaqControl.setTransition - trigger a transition
    #
    def setTransition(self, transition, *, config_alias='BEAM'):
        errorMessage = None
        try:
            msg = create_msg(transition + '.' + config_alias)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            errorMessage = 'setTransition() Exception: %s' % ex
        else:
            try:
                errorMessage = reply['body']['err_info']
            except KeyError:
                pass

        return errorMessage

next_dict = {
    'reset' :       { 'unallocated' : 'plat',
                      'allocated' :   'plat',
                      'connected' :   'plat',
                      'paused' :      'plat',
                      'running' :     'plat' },

    'unallocated' : { 'reset' :       'reset',
                      'allocated' :   'alloc',
                      'connected' :   'alloc',
                      'paused' :      'alloc',
                      'running' :     'alloc' },

    'allocated' :   { 'reset' :       'dealloc',
                      'unallocated' : 'dealloc',
                      'connected' :   'connect',
                      'paused' :      'connect',
                      'running' :     'connect' },

    'connected' :   { 'reset' :       'disconnect',
                      'unallocated' : 'disconnect',
                      'allocated' :   'disconnect',
                      'paused' :      'configure',
                      'running' :     'configure' },

    'paused' :      { 'reset' :       'unconfigure',
                      'unallocated' : 'unconfigure',
                      'allocated' :   'unconfigure',
                      'connected' :   'unconfigure',
                      'running' :     'enable' },

    'running' :     { 'reset' :       'disable',
                      'unallocated' : 'disable',
                      'allocated' :   'disable',
                      'connected' :   'disable',
                      'paused' :      'disable' }
}

def timestampStr():
    current = datetime.now(timezone.utc)
    nsec = 1000 * current.microsecond
    sec = int(current.timestamp()) - POSIX_TIME_AT_EPICS_EPOCH
    return '%010d-%09d' % (sec, nsec)

def create_msg(key, msg_id=None, sender_id=None, body={}):
    if msg_id is None:
        msg_id = timestampStr()
    msg = {'header': {
               'key': key,
               'msg_id': msg_id,
               'sender_id': sender_id},
           'body': body}
    return msg

def error_msg(message):
    body = {'err_info': message}
    return create_msg('error', body=body)

def back_pull_port(platform):
    return PORT_BASE + platform

def back_pub_port(platform):
    return PORT_BASE + platform + 10

def front_rep_port(platform):
    return PORT_BASE + platform + 20

def front_pub_port(platform):
    return PORT_BASE + platform + 30

def get_readout_group_mask(body):
    mask = 0
    if 'drp' in body:
        for key, node_info in body['drp'].items():
            try:
                mask |= (1 << node_info['det_info']['readout'])
            except KeyError:
                pass
    return mask

def wait_for_answers(socket, wait_time, msg_id, err_pub):
    """
    Wait and return all messages from socket that match msg_id
    Parameters
    ----------
    socket: zmq socket
    wait_time: int, wait time in milliseconds
    msg_id: int or None, expected msg_id of received messages
    """
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

        # if key is error then this is an async error message, not a reply
        if msg['header']['key'] == 'error':
            try:
                errMsg = msg['body']['err_info']
                logging.error(errMsg)
                if err_pub is not None:
                    err_pub.send_json(error_msg(errMsg))
            except Exception:
                pass
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


def confirm_response(socket, wait_time, msg_id, ids, err_pub):
    logging.debug('confirm_response(): ids = %s' % ids)
    msgs = []
    for msg in wait_for_answers(socket, wait_time, msg_id, err_pub):
        if msg['header']['sender_id'] in ids:
            msgs.append(msg)
            ids.remove(msg['header']['sender_id'])
            logging.debug('confirm_response(): removed %s from ids' % msg['header']['sender_id'])
        else:
            logging.debug('confirm_response(): %s not in ids' % msg['header']['sender_id'])
        if len(ids) == 0:
            break
    for ii in ids:
        logging.debug('id %s did not respond' % ii)
    return ids, msgs


class CollectionManager():
    def __init__(self, platform, instrument, pv_base, xpm_master, alias, cfg_dbase):
        self.platform = platform
        self.alias = alias
        self.config_alias = None
        self.cfg_dbase = cfg_dbase
        self.xpm_master = xpm_master
        self.pv_base = pv_base
        self.context = zmq.Context(1)
        self.back_pull = self.context.socket(zmq.PULL)
        self.back_pub = self.context.socket(zmq.PUB)
        self.front_rep = self.context.socket(zmq.REP)
        self.front_pub = self.context.socket(zmq.PUB)
        self.back_pull.bind('tcp://*:%d' % back_pull_port(platform))
        self.back_pub.bind('tcp://*:%d' % back_pub_port(platform))
        self.front_rep.bind('tcp://*:%d' % front_rep_port(platform))
        self.front_pub.bind('tcp://*:%d' % front_pub_port(platform))

        # initialize poll set
        self.poller = zmq.Poller()
        self.poller.register(self.back_pull, zmq.POLLIN)
        self.poller.register(self.front_rep, zmq.POLLIN)

        # initialize EPICS context
        self.ctxt = Context('pva')

        # name PVs
        self.pvListMsgHeader = []   # filled in at alloc
        self.pvListXPM = []         # filled in at alloc
        pv_xpm_base = pv_base + ':XPM:%d' % xpm_master
        self.pvGroupL0Enable =  pv_xpm_base+':GroupL0Enable'
        self.pvGroupL0Disable = pv_xpm_base+':GroupL0Disable'
        self.pvGroupMsgInsert = pv_xpm_base+':GroupMsgInsert'

        self.groups = 0     # groups bitmask
        self.cmstate = {}
        self.level_keys = {'drp', 'teb', 'meb', 'control'}
        self.instrument = instrument
        self.ids = set()
        self.handle_request = {
            'selectplatform': self.handle_selectplatform,
            'getinstrument': self.handle_getinstrument,
            'getstate': self.handle_getstate,
            'getstatus': self.handle_getstatus
        }
        self.lastTransition = 'reset'

        self.collectMachine = Machine(self, DaqControl.states, initial='reset', after_state_change='report_status')

        self.collectMachine.add_transition('reset', '*', 'reset',
                                           conditions='condition_reset')
        self.collectMachine.add_transition('plat', ['reset', 'unallocated'], 'unallocated',
                                           conditions='condition_plat')
        self.collectMachine.add_transition('alloc', 'unallocated', 'allocated',
                                           conditions='condition_alloc')
        self.collectMachine.add_transition('dealloc', 'allocated', 'unallocated',
                                           conditions='condition_dealloc')
        self.collectMachine.add_transition('connect', 'allocated', 'connected',
                                           conditions='condition_connect')
        self.collectMachine.add_transition('disconnect', 'connected', 'allocated',
                                           conditions='condition_disconnect')
        self.collectMachine.add_transition('configure', 'connected', 'paused',
                                           conditions='condition_configure')
        self.collectMachine.add_transition('unconfigure', 'paused', 'connected',
                                           conditions='condition_unconfigure')
        self.collectMachine.add_transition('enable', 'paused', 'running',
                                           conditions='condition_enable')
        self.collectMachine.add_transition('disable', 'running', 'paused',
                                           conditions='condition_disable')
        self.collectMachine.add_transition('configupdate', 'paused', 'paused',
                                           conditions='condition_configupdate')

        logging.info('Initial state = %s' % self.state)

        # start main loop
        self.run()

    #
    # cmstate_levels - return copy of cmstate with only drp/teb/meb entries
    #
    def cmstate_levels(self):
        return {k: self.cmstate[k] for k in self.cmstate.keys() & self.level_keys}

    #
    # pv_put -
    #
    def pv_put(self, pvName, val):

        retval = False

        try:
            self.ctxt.put(pvName, val)
        except Exception as ex:
            logging.error("self.ctxt.put('%s', %d) Exception: %s" % (pvName, val, ex))
        else:
            retval = True
            logging.debug("self.ctxt.put('%s', %d)" % (pvName, val))

        return retval

    def service_requests(self):
        # msg['header']['key'] formats:
        #  setstate.STATE.CONFIG_ALIAS
        #  TRANSITION.CONFIG_ALIAS
        #  request
        answer = None
        try:
            msg = self.front_rep.recv_json()
            key = msg['header']['key'].split(".")
            body = msg['body']
            if key[0] == 'setstate':
                self.config_alias = key[2]
                # handle_setstate() sends reply internally
                self.handle_setstate(key[1])
                answer = None
            elif key[0] in DaqControl.transitions:
                self.config_alias = key[1]
                # send 'ok' reply before calling handle_trigger()
                self.front_rep.send_json(create_msg('ok'))
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

    # process asynchronous error reports
    def service_status(self):
        msg = self.back_pull.recv_json()
        try:
            self.report_error(msg['body']['err_info'])
        except Exception:
            pass

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
            self.trigger(key)
        except MachineError as ex:
            logging.debug('MachineError: %s' % ex)
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

        if newstate not in DaqControl.states:
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

    def status_msg(self):
        body = {'state': self.state, 'transition': self.lastTransition,
                'config_alias': str(self.config_alias)}
        return create_msg('status', body=body)

    def report_status(self):
        logging.debug('status: state=%s transition=%s config_alias=%s' %
                      (self.state, self.lastTransition, self.config_alias))
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
        missing, answers = confirm_response(self.back_pull, 5000, None, ids, self.front_pub)
        if missing:
            logging.error('%s phase2 failed' % transition)
            for alias in self.get_aliases(missing):
                self.report_error('%s did not respond to %s' % (alias, transition))
            return False
        return True

    def condition_alloc(self):
        # select procs with active flag set
        ids = self.filter_active_set(self.ids)
        msg = create_msg('alloc', body={'ids': list(ids)})
        self.back_pub.send_multipart([b'all', json.dumps(msg)])

        # make sure all the clients respond to alloc message with their connection info
        retlist, answers = confirm_response(self.back_pull, 1000, msg['header']['msg_id'], ids, self.front_pub)
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

        # assign the readout groups bitmask
        self.groups = get_readout_group_mask(active_state)
        logging.debug('condition_alloc(): groups = 0x%02x' % self.groups)

        # create group-dependent PVs
        self.pvListMsgHeader = []
        self.pvListXPM = []
        for g in range(8):
            if self.groups & (1 << g):
                self.pvListMsgHeader.append(self.pv_base+":PART:"+str(g)+':MsgHeader')
                self.pvListXPM.append(self.pv_base+":PART:"+str(g)+':XPM')
        logging.debug('pvListMsgHeader: %s' % self.pvListMsgHeader)
        logging.debug('pvListXPM: %s' % self.pvListXPM)

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

        logging.debug('cmstate after alloc:\n%s' % self.cmstate)
        self.lastTransition = 'alloc'
        logging.debug('condition_alloc() returning True')
        return True

    def condition_dealloc(self):
        # TODO
        self.lastTransition = 'dealloc'
        logging.debug('condition_dealloc() returning True')
        return True

    def condition_configupdate(self):
        # TODO
        self.lastTransition = 'configupdate'
        logging.debug('condition_configupdate() returning True')
        return True

    def condition_connect(self):
        # set XPM PV
        for pv in self.pvListXPM:
            self.pv_put(pv, self.xpm_master)
        logging.info('master XPM is %d' % self.xpm_master)

        # set Disable PV
        if not self.group_run(False):
            logging.error('condition_connect(): group_run(False) failed')
            return False

        # select procs with active flag set
        ids = self.filter_active_set(self.ids)
        msg = create_msg('connect', body=self.filter_active_dict(self.cmstate_levels()))
        self.back_pub.send_multipart([b'partition', json.dumps(msg)])

        retlist, answers = confirm_response(self.back_pull, 10000, msg['header']['msg_id'], ids, self.front_pub)
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

        retlist, answers = confirm_response(self.back_pull, 30000, msg['header']['msg_id'], ids, self.front_pub)
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

    def handle_getinstrument(self, body):
        logging.debug('handle_getinstrument()')
        body = {'instrument': self.instrument}
        return create_msg('instrument', body=body)

    def handle_selectplatform(self, body):
        logging.debug('handle_selectplatform()')
        if self.state != 'unallocated':
            msg = 'selectPlatform only permitted in unallocated state'
            self.report_error(msg)
            return error_msg(msg)

        try:
            for level, val1 in body.items():
                for key2, val2 in val1.items():
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

    def condition_plat(self):
        self.cmstate.clear()
        self.ids.clear()
        msg = create_msg('plat')
        self.back_pub.send_multipart([b'all', json.dumps(msg)])
        for answer in wait_for_answers(self.back_pull, 1000, msg['header']['msg_id'], self.front_pub):
            for level, item in answer['body'].items():
                if level not in self.cmstate:
                    self.cmstate[level] = {}
                id = answer['header']['sender_id']
                self.cmstate[level][id] = item
                self.cmstate[level][id]['active'] = DaqControl.default_active
                if level == 'drp':
                    if 'det_info' not in self.cmstate[level][id]:
                        self.cmstate[level][id]['det_info'] = {}
                    self.cmstate[level][id]['det_info']['readout'] = self.platform
                self.ids.add(id)
        if len(self.ids) == 0:
            self.report_error('no clients responded to plat')
            retval = False
        else:
            retval = True
            self.lastTransition = 'plat'

        # add control info
        if not 'control' in self.cmstate:
            self.cmstate['control'] = {}
            self.cmstate['control'][0] = {}
            self.cmstate['control'][0]['active'] = 1
            self.cmstate['control'][0]['control_info'] = {}
            self.cmstate['control'][0]['proc_info'] = {}
            self.cmstate['control'][0]['control_info']['xpm_master'] = self.xpm_master
            self.cmstate['control'][0]['control_info']['pv_base'] = self.pv_base
            self.cmstate['control'][0]['control_info']['cfg_dbase'] = self.cfg_dbase
            self.cmstate['control'][0]['control_info']['instrument'] = self.instrument
            self.cmstate['control'][0]['proc_info']['alias'] = self.alias
            self.cmstate['control'][0]['proc_info']['host'] = socket.gethostname()
            self.cmstate['control'][0]['proc_info']['pid'] = os.getpid()

        logging.debug('cmstate after plat:\n%s' % self.cmstate)
        logging.debug('condition_plat() returning %s' % retval)
        return retval

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

    def condition_common(self, transition, timeout, body={}):
        retval = True
        # select procs with active flag set
        ids = self.filter_active_set(self.ids)
        msg = create_msg(transition, body=body)
        self.back_pub.send_multipart([b'partition', json.dumps(msg)])

        # only drp and teb groups (aka levels) respond to configure and above
        ids = self.filter_level('drp', ids) | self.filter_level('teb', ids)

        if len(ids) == 0:
            logging.debug('condition_common() empty set of ids')
            return True

        # make sure all the clients respond to transition before timeout
        retlist, answers = confirm_response(self.back_pull, timeout, msg['header']['msg_id'], ids, self.front_pub)
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
        # phase 1
        ok = self.condition_common('configure', 15000,
                                   body={'config_alias': self.config_alias})
        if not ok:
            logging.error('condition_configure(): configure phase1 failed')
            return False

        # phase 2
        # ...clear readout
        for pv in self.pvListMsgHeader:
            self.pv_put(pv, DaqControl.transitionId['ClearReadout'])
        self.pv_put(self.pvGroupMsgInsert, self.groups)
        self.pv_put(self.pvGroupMsgInsert, 0)
        # ...configure
        for pv in self.pvListMsgHeader:
            self.pv_put(pv, DaqControl.transitionId['Configure'])
        self.pv_put(self.pvGroupMsgInsert, self.groups)
        self.pv_put(self.pvGroupMsgInsert, 0)

        ok = self.get_phase2_replies('configure')
        if not ok:
            return False

        logging.debug('condition_configure() returning %s' % ok)

        self.lastTransition = 'configure'
        return True

    def condition_unconfigure(self):
        # phase 1
        ok = self.condition_common('unconfigure', 6000)
        if not ok:
            logging.error('condition_unconfigure(): unconfigure phase1 failed')
            return False

        # phase 2
        for pv in self.pvListMsgHeader:
            self.pv_put(pv, DaqControl.transitionId['Unconfigure'])
        self.pv_put(self.pvGroupMsgInsert, self.groups)
        self.pv_put(self.pvGroupMsgInsert, 0)

        ok = self.get_phase2_replies('unconfigure')
        if not ok:
            return False

        logging.debug('condition_unconfigure() returning %s' % ok)

        self.lastTransition = 'unconfigure'
        return True

    def group_run(self, enable):
        if enable:
            rv = self.pv_put(self.pvGroupL0Enable, self.groups)
        else:
            rv = self.pv_put(self.pvGroupL0Disable, self.groups)
        return rv

    def condition_enable(self):
        # phase 1
        ok = self.condition_common('enable', 6000)
        if not ok:
            logging.error('condition_enable(): enable phase1 failed')
            return False

        # phase 2
        for pv in self.pvListMsgHeader:
            self.pv_put(pv, DaqControl.transitionId['Enable'])
        self.pv_put(self.pvGroupMsgInsert, self.groups)
        self.pv_put(self.pvGroupMsgInsert, 0)

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
            logging.error('condition_enable(): group_run(False) failed')
            return False

        # phase 1
        ok = self.condition_common('disable', 6000)
        if not ok:
            logging.error('condition_disable(): disable phase1 failed')
            return False

        # phase 2
        for pv in self.pvListMsgHeader:
            self.pv_put(pv, DaqControl.transitionId['Disable'])
        self.pv_put(self.pvGroupMsgInsert, self.groups)
        self.pv_put(self.pvGroupMsgInsert, 0)

        ok = self.get_phase2_replies('disable')
        if not ok:
            return False

        self.lastTransition = 'disable'
        return True


    def condition_reset(self):
        # is a reply to reset necessary?
        msg = create_msg('reset')
        self.back_pub.send_multipart([b'all', json.dumps(msg)])
        self.lastTransition = 'reset'
        logging.debug('condition_reset() returning True')
        return True

class Client:
    def __init__(self, platform):
        self.context = zmq.Context(1)
        self.back_push = self.context.socket(zmq.PUSH)
        self.back_sub = self.context.socket(zmq.SUB)
        self.back_push.connect('tcp://localhost:%d' % back_pull_port(platform))
        self.back_sub.connect('tcp://localhost:%d' % back_pub_port(platform))
        self.back_sub.setsockopt(zmq.SUBSCRIBE, b'')
        handle_request = {
            'plat': self.handle_plat,
            'alloc': self.handle_alloc,
            'connect': self.handle_connect
        }
        while True:
            try:
                msg = self.back_sub.recv_json()
                key = msg['header']['key']
                handle_request[key](msg)
            except KeyError as ex:
                logging.debug('KeyError: %s' % ex)

            if key == 'connect':
                break

    def handle_plat(self, msg):
        logging.debug('Client handle_plat()')
        # time.sleep(1.5)
        hostname = socket.gethostname()
        pid = os.getpid()
        self.id = hash(hostname+str(pid))
        body = {'drp': {'proc_info': {
                        'host': hostname,
                        'pid': pid}}}
        reply = create_msg('plat', msg['header']['msg_id'], self.id, body=body)
        self.back_push.send_json(reply)

    def handle_alloc(self, msg):
        logging.debug('Client handle_alloc()')
        body = {'drp': {'connect_info': {'infiniband': '123.456.789'}}}
        reply = create_msg('alloc', msg['header']['msg_id'], self.id, body)
        self.back_push.send_json(reply)
        self.state = 'alloc'

    def handle_connect(self, msg):
        logging.debug('Client handle_connect()')
        if self.state == 'alloc':
            reply = create_msg('ok', msg['header']['msg_id'], self.id)
            self.back_push.send_json(reply)


def main():
    from multiprocessing import Process

    # Process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, choices=range(0, 8), default=0, help='platform (default 0)')
    parser.add_argument('-x', metavar='XPM', type=int, required=True, help='master XPM')
    parser.add_argument('-P', metavar='INSTRUMENT', default='TST', help='instrument (default TST)')
    parser.add_argument('-d', metavar='CFGDATABASE', default='mcbrowne:psana@psdb-dev:9306/configDB', help='configuration database connection')
    parser.add_argument('-B', metavar='PVBASE', required=True, help='PV base')
    parser.add_argument('-u', metavar='ALIAS', required=True, help='unique ID')
    parser.add_argument('-a', action='store_true', help='autoconnect')
    parser.add_argument('-v', action='store_true', help='be verbose')
    args = parser.parse_args()
    platform = args.p

    if args.v:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    def manager():
        manager = CollectionManager(platform, args.P, args.B, args.x, args.u, args.d)

    def client(i):
        c = Client(platform)

    procs = [Process(target=manager)]
    for i in range(2):
        # procs.append(Process(target=client, args=(i,)))
        pass

    for p in procs:
        p.start()
        pass

    if args.a:
        # Commands
        context = zmq.Context(1)
        front_req = context.socket(zmq.REQ)
        front_req.connect('tcp://localhost:%d' % front_rep_port(platform))
        time.sleep(0.5)

        msg = create_msg('plat')
        front_req.send_json(msg)
        print('Answer to plat:', front_req.recv_multipart())

        msg = create_msg('alloc')
        front_req.send_json(msg)
        print('Answer to alloc:', front_req.recv_multipart())

        msg = create_msg('connect')
        front_req.send_json(msg)
        print('Answer to connect:', front_req.recv_multipart())

    for p in procs:
        try:
            p.join()
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
