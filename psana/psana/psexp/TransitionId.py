ClearReadout = 0
Reset = 1
Configure = 2
Unconfigure = 3
BeginRun = 4
EndRun = 5
BeginStep = 6
EndStep = 7
Enable = 8
Disable = 9
SlowUpdate = 10
L1Accept_EndOfBatch = 11
L1Accept = 12
NumberOf = 13


def isEvent(transition_id):
    if transition_id == L1Accept or transition_id == L1Accept_EndOfBatch:
        return True
    return False
