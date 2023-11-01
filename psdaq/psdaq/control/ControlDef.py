from datetime import datetime, timezone

class ControlDef:

    # transitionId is a subset of the TransitionId.hh enum
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
        if type(value) == str:
            self.position = value
        elif type(value) == int:
            self.position = "step%d" % value

def timestampStr():
    current = datetime.now(timezone.utc)
    nsec = 1000 * current.microsecond
    sec = int(current.timestamp()) - ControlDef.POSIX_TIME_AT_EPICS_EPOCH
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

def warning_msg(message):
    body = {'err_info': message}
    return create_msg('warning', body=body)

def fileReport_msg(path):
    body = {'path': path}
    return create_msg('fileReport', body=body)

def chunkRequest_msg(offset):
    body = {'offset': offset}
    return create_msg('chunkRequest', body=body)

def progress_msg(transition, elapsed, total):
    body = {'transition': transition, 'elapsed': int(elapsed), 'total': int(total)}
    return create_msg('progress', body=body)

def step_msg(doneFlag):
    body = {'step_done': doneFlag}
    return create_msg('step', body=body)

def back_pull_port(platform):
    return ControlDef.PORT_BASE + platform

def back_pub_port(platform):
    return ControlDef.PORT_BASE + platform + 10

def front_rep_port(platform):
    return ControlDef.PORT_BASE + platform + 20

def front_pub_port(platform):
    return ControlDef.PORT_BASE + platform + 30

def fast_rep_port(platform):
    return ControlDef.PORT_BASE + platform + 40

def step_pub_port(platform):
    return ControlDef.PORT_BASE + platform + 50

def scan_pull_port(platform):
    return ControlDef.PORT_BASE + platform + 60
