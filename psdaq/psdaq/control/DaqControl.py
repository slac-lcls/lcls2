"""Client-side `DaqControl` class that talks to the DAQ collection process over ZMQ.

Requests go over REQ sockets to `front_rep_port` and `fast_rep_port`; status updates
are received on a SUB socket connected to `front_pub_port` and `step_pub_port`.
"""
import logging
import zmq
from psdaq.control.ControlDef import ControlDef, front_pub_port, front_rep_port, fast_rep_port, create_msg
from psdaq.control.ControlDef import step_pub_port

class DaqControl:
    'Base class for controlling data acquisition'

    # default readout group is self.platform

    def __init__(self, *, host, platform, timeout):
        self.host = host
        self.platform = platform
        self.timeout = timeout

        # initialize zmq socket
        self.context = zmq.Context(1)
        self.front_sub = self.context.socket(zmq.SUB)
        self.front_sub.connect('tcp://%s:%d' % (host, front_pub_port(platform)))
        self.front_sub.connect('tcp://%s:%d' % (host, step_pub_port(platform)))
        self.front_sub.setsockopt(zmq.SUBSCRIBE, b'')
        self.front_req = None
        self.front_req_endpoint = 'tcp://%s:%d' % (host, front_rep_port(platform))
        self.front_req_init()
        self.fast_req = None
        self.fast_req_endpoint = 'tcp://%s:%d' % (host, fast_rep_port(platform))
        self.fast_req_init()

    #
    # DaqControl.getState - get current state
    #
    def getState(self):
        """Send a 'getstate' request and return the reply's ``header['key']``.

        On a receive timeout (zmq.Again) the front REQ socket is re-created.

        Returns
        -------
        str
            The reply key, or 'error' on timeout, exception, KeyboardInterrupt or a missing key.
        """
        retval = 'error'
        try:
            msg = create_msg('getstate')
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except zmq.Again:
            logging.error('getState(%s) timeout (%.1f sec)' % (self.front_req_endpoint, self.timeout / 1000.))
            logging.info('getState() reinitializing zmq socket')
            self.front_req_init()
        except Exception as ex:
            logging.error('getState(%s) Exception: %s' % (self.front_req_endpoint, ex))
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
        """Send a 'getstate' request and return the reply's body.

        On a receive timeout (zmq.Again) the front REQ socket is re-created.

        Returns
        -------
        dict
            ``reply['body']``, or {} on timeout, exception, KeyboardInterrupt or a missing key.
        """
        retval = {}
        try:
            msg = create_msg('getstate')
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except zmq.Again:
            logging.error('getPlatform(%s) timeout (%.1f sec)' % (self.front_req_endpoint, self.timeout / 1000.))
            logging.info('getPlatform() reinitializing zmq socket')
            self.front_req_init()
        except Exception as ex:
            logging.error('getPlatform(%s) Exception: %s' % (self.front_req_endpoint, ex))
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        else:
            try:
                retval = reply['body']
            except KeyError:
                pass

        return retval

    #
    # DaqControl.getJsonConfig - get json configuration
    #
    def getJsonConfig(self):
        """Return the platform info converted by `levels_to_activedet` as indented, sorted JSON.

        Note that neither `levels_to_activedet` nor `oldjson` is imported in this module, so
        as written this raises NameError.
        """
        src = self.getPlatform()
        dst = levels_to_activedet(src)
        retval =  oldjson.dumps(dst, sort_keys=True, indent=4)
        logging.debug('getJsonConfig() return value: %s' % retval)
        return retval

    #
    # DaqControl.storeJsonConfig - store json configuration
    #
    def storeJsonConfig(self, json_data):
        """Send a 'storejsonconfig' request with body ``{'json_data': json_data}``.

        On a receive timeout (zmq.Again) the front REQ socket is re-created.

        Parameters
        ----------
        json_data
            Data placed in the request body.

        Returns
        -------
        dict
            ``reply['body']``, or {} on timeout, exception, KeyboardInterrupt or a missing key.
        """
        retval = {}
        body = {"json_data": json_data}
        try:
            msg = create_msg('storejsonconfig', body=body)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except zmq.Again:
            logging.error('storeJsonConfig(%s) timeout (%.1f sec)' % (self.front_req_endpoint, self.timeout / 1000.))
            logging.info('storeJsonConfig() reinitializing zmq socket')
            self.front_req_init()
        except Exception as ex:
            logging.error('storeJsonConfig(%s) Exception: %s' % (self.front_req_endpoint, ex))
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
        """Send a 'selectplatform' request with the given body.

        On a receive timeout (zmq.Again) the front REQ socket is re-created.

        Parameters
        ----------
        body : dict
            Request body.

        Returns
        -------
        dict
            ``reply['body']``, or {} on timeout, exception, KeyboardInterrupt or a missing key.
        """
        retval = {}
        try:
            msg = create_msg('selectplatform', body=body)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except zmq.Again:
            logging.error('selectPlatform(%s) timeout (%.1f sec)' % (self.front_req_endpoint, self.timeout / 1000.))
            logging.info('selectPlatform() reinitializing zmq socket')
            self.front_req_init()
        except Exception as ex:
            logging.error('selectPlatform(%s) Exception: %s' % (self.front_req_endpoint, ex))
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
        """Send a 'getinstrument' request on the fast REQ socket.

        Returns
        -------
        str or None
            ``reply['body']['instrument']``, or None if the request or lookup fails (the
            exception is printed).
        """
        r1 = None
        try:
            msg = create_msg('getinstrument')
            self.fast_req.send_json(msg)
            reply = self.fast_req.recv_json()
        except Exception as ex:
            print('getInstrument(%s) Exception: %s' % (self.fast_req_endpoint, ex))
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
        """Send a 'getstatus' request and return fields of the reply body.

        Returns
        -------
        tuple
            (transition, state, config_alias, recording, platform, bypass_activedet,
            experiment_name, run_number, last_run_number). Defaults are 'error' for every
            field except platform ({}); on a KeyError, fields read before it keep their values.
        """
        r1 = r2 = r3 = r4 = r6 = 'error'
        r5 = {}
        r7 = r8 = r9 = 'error'
        try:
            msg = create_msg('getstatus')
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            print('getStatus(%s) Exception: %s' % (self.front_req_endpoint, ex))
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        else:
            try:
                r1 = reply['body']['transition']
                r2 = reply['body']['state']
                r3 = reply['body']['config_alias']
                r4 = reply['body']['recording']
                r5 = reply['body']['platform']
                r6 = reply['body']['bypass_activedet']
                r7 = reply['body']['experiment_name']
                r8 = reply['body']['run_number']
                r9 = reply['body']['last_run_number']
            except KeyError:
                pass

        return (r1, r2, r3, r4, r5, r6, r7, r8, r9)

    #
    # DaqControl.monitorStatus - monitor the status
    #
    def monitorStatus(self):

        # process messages
        """Block on the SUB socket until a recognized message arrives and return its fields.

        Messages with other keys are ignored. Always returns an 8-tuple:
        'status' -> (transition, state, config_alias, recording, bypass_activedet,
        experiment_name, run_number, last_run_number); 'error'/'warning' -> (key, err_info,
        'error' x6); 'fileReport' -> ('fileReport', path, 'error' x6); 'progress' ->
        ('progress', transition, elapsed, total, 'error' x4); 'step' -> ('step', step_done,
        'error' x6). On KeyboardInterrupt or KeyError returns eight Nones.
        """
        while True:
            try:
                msg = self.front_sub.recv_json()

                if msg['header']['key'] == 'status':
                    # return transition, state, config_alias, recording, bypass_activedet, experiment_name, run_number, last_run_number
                    return msg['body']['transition'], msg['body']['state'], msg['body']['config_alias'], msg['body']['recording'], msg['body']['bypass_activedet'],\
                           msg['body']['experiment_name'], msg['body']['run_number'], msg['body']['last_run_number']

                elif msg['header']['key'] == 'error':
                    # return 'error', error message, 'error', 'error', 'error', 'error', 'error', 'error'
                    return 'error', msg['body']['err_info'], 'error', 'error', 'error', 'error', 'error', 'error'

                elif msg['header']['key'] == 'warning':
                    # return 'error', error message, 'error', 'error', 'error', 'error', 'error', 'error'
                    return 'warning', msg['body']['err_info'], 'error', 'error', 'error', 'error', 'error', 'error'

                elif msg['header']['key'] == 'fileReport':
                    # return 'fileReport', path, 'error', 'error', 'error', 'error', 'error', 'error'
                    return 'fileReport', msg['body']['path'], 'error', 'error', 'error', 'error', 'error', 'error'

                elif msg['header']['key'] == 'progress':
                    # return 'progress', transition, elapsed, total, 'error', 'error', 'error', 'error'
                    return 'progress', msg['body']['transition'], msg['body']['elapsed'], msg['body']['total'], 'error', 'error', 'error', 'error'

                elif msg['header']['key'] == 'step':
                    # return 'step', step_done, 'error', 'error', 'error', 'error', 'error', 'error'
                    return 'step', msg['body']['step_done'], 'error', 'error', 'error', 'error', 'error', 'error'

            except KeyboardInterrupt:
                break

            except KeyError as ex:
                logging.error('KeyError: %s' % ex)
                break

        return None, None, None, None, None, None, None, None

    #
    # DaqControl.setState - change the state
    # The optional second argument is a dictionary containing
    # one entry per transition that contains information that
    # will be put into the phase1-json of the transition. An example:
    # {'beginstep': {'myvalue1':3 , 'myvalue2': {'myvalue3':72}},
    #  'enable':    {'myvalue5':37, 'myvalue6': 'hello'}}
    #
    def setState(self, state, phase1Info={}):
        """Send a 'setstate.<state>' request with `phase1Info` as the body.

        On a receive timeout (zmq.Again) the front REQ socket is re-created.

        Parameters
        ----------
        state : str
            Target state name.
        phase1Info : dict, optional
            Per-transition phase-1 info, e.g. ``{'beginstep': {...}, 'enable': {...}}``.

        Returns
        -------
        str or None
            ``reply['body']['err_info']`` if present, an error string on timeout/exception, else None.
        """
        errorMessage = None
        try:
            msg = create_msg('setstate.' + state, body=phase1Info)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except zmq.Again:
            errorMessage = 'setState(%s) timeout (%.1f sec)' % (self.front_req_endpoint, self.timeout / 1000.)
            logging.info('setState() reinitializing zmq socket')
            self.front_req_init()
        except Exception as ex:
            errorMessage = 'setState(%s) Exception: %s' % (self.front_req_endpoint, ex)
        else:
            try:
                errorMessage = reply['body']['err_info']
            except KeyError:
                pass

        return errorMessage

    #
    # DaqControl.setConfig - set BEAM/NOBEAM
    #
    def setConfig(self, config):
        """Send a 'setconfig.<config>' request on the front REQ socket.

        Returns
        -------
        str or None
            ``reply['body']['err_info']`` if present, an error string on exception, else None.
        """
        errorMessage = None
        try:
            msg = create_msg('setconfig.' + config)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            errorMessage = 'setConfig(%s) Exception: %s' % (self.front_req_endpoint, ex)
        else:
            try:
                errorMessage = reply['body']['err_info']
            except KeyError:
                pass

        return errorMessage

    #
    # DaqControl.setRecord - set record flag
    #   True or False
    #
    def setRecord(self, recordIn):
        """Send 'setrecord.1' or 'setrecord.0' on the fast REQ socket.

        Parameters
        ----------
        recordIn : bool
            Record flag; any non-bool value is rejected without sending.

        Returns
        -------
        str or None
            ``reply['body']['err_info']`` if present, an error string on exception or non-bool
            input, else None.
        """
        errorMessage = None
        if type(recordIn) == type(True):
            if recordIn:
                record = '1'
            else:
                record = '0'

            try:
                msg = create_msg('setrecord.' + record)
                self.fast_req.send_json(msg)
                reply = self.fast_req.recv_json()
            except Exception as ex:
                errorMessage = 'setRecord(%s) Exception: %s' % (self.fast_req_endpoint, ex)
            else:
                try:
                    errorMessage = reply['body']['err_info']
                except KeyError:
                    pass
        else:
            errorMessage = 'setRecord() requires True or False'

        return errorMessage

    #
    # DaqControl.getBlock -
    #
    def getBlock(self, data):
        """Send a 'getblock' request with `data` as body on the fast REQ socket.

        Returns
        -------
        object or None
            ``reply['body']``, or None if the request or lookup fails (the exception is printed).
        """
        r1 = None
        try:
            msg = create_msg('getblock', body=data)
            self.fast_req.send_json(msg)
            reply = self.fast_req.recv_json()
            logging.debug(f'getBlock reply={reply}')
        except Exception as ex:
            print('getBlock(%s) Exception 1: %s' % (self.fast_req_endpoint, ex))
        else:
            try:
                r1 = reply['body']
            except Exception as ex:
                print('getBlock() Exception 2: %s' % ex)

        return r1

    #
    # DaqControl.setBypass - set bypass_activedet flag
    #   True or False
    #
    def setBypass(self, bypassIn):
        """Send 'setbypass.1' or 'setbypass.0' on the front REQ socket.

        Parameters
        ----------
        bypassIn : bool
            Bypass flag; any non-bool value is rejected without sending.

        Returns
        -------
        str or None
            ``reply['body']['err_info']`` if present, an error string on exception or non-bool
            input, else None.
        """
        errorMessage = None
        if type(bypassIn) == type(True):
            if bypassIn:
                bypass = '1'
            else:
                bypass = '0'

            try:
                msg = create_msg('setbypass.' + bypass)
                self.front_req.send_json(msg)
                reply = self.front_req.recv_json()
            except Exception as ex:
                errorMessage = 'setBypass(%s) Exception: %s' % (self.front_req_endpoint, ex)
            else:
                try:
                    errorMessage = reply['body']['err_info']
                except KeyError:
                    pass
        else:
            errorMessage = 'setBypass() requires True or False'

        return errorMessage

    #
    # DaqControl.setTransition - trigger a transition
    # The optional second argument is a dictionary containing
    # information that will be put into the phase1-json of the transition.
    # An example:
    # {'myvalue1':3 , 'myvalue2': {'myvalue3':72}}
    #
    def setTransition(self, transition, phase1Info={}):
        """Send a request whose key is `transition` with `phase1Info` as the body.

        Returns
        -------
        str or None
            ``reply['body']['err_info']`` if present, an error string on exception, else None.
        """
        errorMessage = None
        try:
            msg = create_msg(transition, body=phase1Info)
            self.front_req.send_json(msg)
            reply = self.front_req.recv_json()
        except Exception as ex:
            errorMessage = 'setTransition(%s) Exception: %s' % (self.front_req_endpoint, ex)
        else:
            try:
                errorMessage = reply['body']['err_info']
            except KeyError:
                pass

        return errorMessage

    #
    # DaqControl.front_req_init - (re)initialize the front_req zmq socket
    #
    def front_req_init(self):
        # if socket previouly created, close it
        """Close the existing front REQ socket (if any) and create a new one.

        The new socket has linger 0 and RCVTIMEO set to `self.timeout`, and is connected to
        `front_req_endpoint`.
        """
        if self.front_req is not None:
            self.front_req.close()
        # create new socket
        self.front_req = self.context.socket(zmq.REQ)
        self.front_req.linger = 0
        self.front_req.RCVTIMEO = self.timeout
        self.front_req.connect(self.front_req_endpoint)

    #
    # DaqControl.fast_req_init - (re)initialize the fast_req zmq socket
    #
    def fast_req_init(self):
        # if socket previouly created, close it
        """Close the existing fast REQ socket (if any) and create a new one.

        The new socket has linger 0 and RCVTIMEO set to `self.timeout`, and is connected to
        `fast_req_endpoint`.
        """
        if self.fast_req is not None:
            self.fast_req.close()
        # create new socket
        self.fast_req = self.context.socket(zmq.REQ)
        self.fast_req.linger = 0
        self.fast_req.RCVTIMEO = self.timeout
        self.fast_req.connect(self.fast_req_endpoint)
