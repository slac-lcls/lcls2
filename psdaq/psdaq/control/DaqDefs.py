from datetime import datetime, timezone

class DaqDefs:

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


    PORT_BASE = 29980
    POSIX_TIME_AT_EPICS_EPOCH = 631152000

def timestampStr():
    current = datetime.now(timezone.utc)
    nsec = 1000 * current.microsecond
    sec = int(current.timestamp()) - DaqDefs.POSIX_TIME_AT_EPICS_EPOCH
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

def progress_msg(transition, elapsed, total):
    body = {'transition': transition, 'elapsed': int(elapsed), 'total': int(total)}
    return create_msg('progress', body=body)

def back_pull_port(platform):
    return DaqDefs.PORT_BASE + platform

def back_pub_port(platform):
    return DaqDefs.PORT_BASE + platform + 10

def front_rep_port(platform):
    return DaqDefs.PORT_BASE + platform + 20

def front_pub_port(platform):
    return DaqDefs.PORT_BASE + platform + 30

