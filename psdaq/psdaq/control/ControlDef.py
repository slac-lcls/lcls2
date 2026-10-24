"""Shared constants and helpers for the DAQ control system.

Defines the `ControlDef` constants class, two fake PV classes, message builders that
return the `{'header': ..., 'body': ...}` dicts used by the control messages, and
functions that compute ZMQ port numbers from a platform number or XPM name.
"""
from datetime import datetime, timezone

class ControlDef:

    # transitionId is a subset of the TransitionId.hh enum
    """Namespace of DAQ control constants.

    Holds the `transitionId` map (commented in the code as "a subset of the TransitionId.hh
    enum"), the list of `transitions` and `states` names, `CHUNKINFO`/`STEPINFO` values,
    `PORT_BASE` (29980), `POSIX_TIME_AT_EPICS_EPOCH` and `STEP_VALUE` ('step_value').
    """
    transitionId = {
        'ClearReadout'      : 0,
        'Reset'             : 1,
        'Configure'         : 2,
        'Unconfigure'       : 3,
        'BeginRun'          : 4,
        'EndRun'            : 5,
        'BeginStep'         : 6,
        'EndStep'           : 7,
        'Enable'            : 8,
        'Disable'           : 9,
        'SlowUpdate'        : 10,
        'L1Accept'          : 12,
    }

    transitions = ['rollcall', 'alloc', 'dealloc',
                   'connect', 'disconnect',
                   'configure', 'unconfigure',
                   'beginrun', 'endrun',
                   'beginstep', 'endstep',
                   'enable', 'disable',
                   'slowupdate', 'reset']

    states = [
        'reset',
        'unallocated',
        'allocated',
        'connected',
        'configured',
        'starting',
        'paused',
        'running'
    ]

    CHUNKINFO = 252         # psdaq/drp/drp.hh
    STEPINFO = 253          # psdaq/drp/drp.hh
    PORT_BASE = 29980
    POSIX_TIME_AT_EPICS_EPOCH = 631152000
    #  name of simulated motor reserved for step value
    STEP_VALUE = 'step_value'

class MyFloatPv:
    """Fake float PV"""
    def __init__(self, name):
        self.name = name
        self.position = 0.0

    def update(self, value):
        """Set `position` from `value` if it is a float or int.

        An int is converted to float; values of any other type are ignored.

        Parameters
        ----------
        value : float or int
            New position value.
        """
        if type(value) == float:
            self.position = value
        elif type(value) == int:
            self.position = float(value)

class MyStringPv:
    """Fake string PV"""
    def __init__(self, name):
        self.name = name
        self.position = "step0"

    def update(self, value):
        """Set `position` from `value` if it is a str or int.

        A str is stored as-is; an int is stored as ``"step%d" % value``. Values of any other
        type are ignored.

        Parameters
        ----------
        value : str or int
            New position value.
        """
        if type(value) == str:
            self.position = value
        elif type(value) == int:
            self.position = "step%d" % value

def timestampStr():
    """Return the current UTC time as an EPICS-epoch timestamp string.

    Seconds are counted from the EPICS epoch (POSIX time minus
    `ControlDef.POSIX_TIME_AT_EPICS_EPOCH`); nanoseconds come from the microsecond field.

    Returns
    -------
    str
        String formatted as ``'%010d-%09d' % (sec, nsec)``.
    """
    current = datetime.now(timezone.utc)
    nsec = 1000 * current.microsecond
    sec = int(current.timestamp()) - ControlDef.POSIX_TIME_AT_EPICS_EPOCH
    return '%010d-%09d' % (sec, nsec)

def create_msg(key, msg_id=None, sender_id=None, body={}):
    """Build a control message dict.

    Parameters
    ----------
    key : str
        Value for ``header['key']``.
    msg_id : str, optional
        Value for ``header['msg_id']``; if None, `timestampStr()` is used.
    sender_id : optional
        Value for ``header['sender_id']``.
    body : dict, optional
        Value for ``'body'``; default is a shared empty dict.

    Returns
    -------
    dict
        ``{'header': {'key': key, 'msg_id': msg_id, 'sender_id': sender_id}, 'body': body}``.
    """
    if msg_id is None:
        msg_id = timestampStr()
    msg = {'header': {
               'key': key,
               'msg_id': msg_id,
               'sender_id': sender_id},
           'body': body}
    return msg

def error_msg(message):
    """Return a message with key 'error' and body ``{'err_info': message}``.

    Parameters
    ----------
    message : str
        Error text.

    Returns
    -------
    dict
        Message built by `create_msg`.
    """
    body = {'err_info': message}
    return create_msg('error', body=body)

def warning_msg(message):
    """Return a message with key 'warning' and body ``{'err_info': message}``.

    Parameters
    ----------
    message : str
        Warning text.

    Returns
    -------
    dict
        Message built by `create_msg`.
    """
    body = {'err_info': message}
    return create_msg('warning', body=body)

def fileReport_msg(path):
    """Return a message with key 'fileReport' and body ``{'path': path}``.

    Parameters
    ----------
    path : str
        File path to report.

    Returns
    -------
    dict
        Message built by `create_msg`.
    """
    body = {'path': path}
    return create_msg('fileReport', body=body)

def chunkRequest_msg(offset):
    """Return a message with key 'chunkRequest' and body ``{'offset': offset}``.

    Parameters
    ----------
    offset
        Value stored under ``'offset'``.

    Returns
    -------
    dict
        Message built by `create_msg`.
    """
    body = {'offset': offset}
    return create_msg('chunkRequest', body=body)

def progress_msg(transition, elapsed, total):
    """Return a message with key 'progress'.

    The body is ``{'transition': transition, 'elapsed': int(elapsed), 'total': int(total)}``.

    Parameters
    ----------
    transition : str
        Transition name.
    elapsed : number
        Elapsed amount, truncated to int.
    total : number
        Total amount, truncated to int.

    Returns
    -------
    dict
        Message built by `create_msg`.
    """
    body = {'transition': transition, 'elapsed': int(elapsed), 'total': int(total)}
    return create_msg('progress', body=body)

def step_msg(doneFlag):
    """Return a message with key 'step' and body ``{'step_done': doneFlag}``.

    Parameters
    ----------
    doneFlag
        Value stored under ``'step_done'``.

    Returns
    -------
    dict
        Message built by `create_msg`.
    """
    body = {'step_done': doneFlag}
    return create_msg('step', body=body)

def back_pull_port(platform):
    """Return ``ControlDef.PORT_BASE + platform``."""
    return ControlDef.PORT_BASE + platform

def back_pub_port(platform):
    """Return ``ControlDef.PORT_BASE + platform + 10``."""
    return ControlDef.PORT_BASE + platform + 10

def front_rep_port(platform):
    """Return ``ControlDef.PORT_BASE + platform + 20``."""
    return ControlDef.PORT_BASE + platform + 20

def front_pub_port(platform):
    """Return ``ControlDef.PORT_BASE + platform + 30``."""
    return ControlDef.PORT_BASE + platform + 30

def fast_rep_port(platform):
    """Return ``ControlDef.PORT_BASE + platform + 40``."""
    return ControlDef.PORT_BASE + platform + 40

def step_pub_port(platform):
    """Return ``ControlDef.PORT_BASE + platform + 50``."""
    return ControlDef.PORT_BASE + platform + 50

def scan_pull_port(platform):
    """Return ``ControlDef.PORT_BASE + platform + 60``."""
    return ControlDef.PORT_BASE + platform + 60

def xpm_pull_port(xpm_name):
    #  Extract a unique integer from the xpm name
    #  NEH,FEH use the same host
    """Return a port number derived from an XPM name.

    The integer after the last ':' is taken as the XPM number; 16 is added if the second
    ':'-separated field is 'FEH'. The result is ``ControlDef.PORT_BASE + xpm + 70``.

    Parameters
    ----------
    xpm_name : str
        XPM name with ':'-separated fields, e.g. 'DAQ:NEH:XPM:0'.

    Returns
    -------
    int
        Port number.
    """
    xpm = int(xpm_name.rsplit(':',1)[1])
    if xpm_name.split(':',2)[1]=='FEH':
        xpm += 16
    return ControlDef.PORT_BASE + xpm + 70
