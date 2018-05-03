"""
ControlState - Control State Machine Class

Author: Chris Ford <caf@slac.stanford.edu>
"""

class ControlState(object):

    def __init__(self, key):
        self._key = key
        self._transition_dict = {}

    def __repr__(self): return self.__str__()

    def __str__(self): return self._key.decode()

    def key(self): return self._key

    def register(self, transition_dict):
        self._transition_dict = transition_dict

    def on_transition(self, transition):

        # In case the transition func is not found, or it returns False,
        # return the current state: self.
        retval = self

        if transition in self._transition_dict:
            dofunc, nextstate = self._transition_dict[transition]
            if dofunc():
                # Transition func returned True -- return new state
                retval = nextstate
        return retval


class TestStateMachine(object):

    # Define transitions

    transition_configure = b'CONFIGURE'
    transition_unconfigure = b'UNCONFIGURE'
    transition_beginrun = b'BEGINRUN'
    transition_endrun = b'ENDRUN'
    transition_enable = b'ENABLE'
    transition_disable = b'DISABLE'

    # Define states

    state_unconfigured = ControlState(b'UNCONFIGURED')
    state_configured = ControlState(b'CONFIGURED')
    state_running = ControlState(b'RUNNING')
    state_enabled = ControlState(b'ENABLED')

    def __init__(self):

        # register callbacks for each valid state+transition combination

        unconfigured_dict = {
            self.transition_configure: (self.configfunc, self.state_configured)
        }

        configured_dict = {
            self.transition_unconfigure: (self.unconfigfunc,
                                          self.state_unconfigured),
            self.transition_beginrun: (self.beginrunfunc, self.state_running)
        }

        running_dict = {
            self.transition_endrun: (self.endrunfunc, self.state_configured),
            self.transition_enable: (self.enablefunc, self.state_enabled)
        }

        enabled_dict = {
            self.transition_disable: (self.disablefunc, self.state_running)
        }

        self.state_unconfigured.register(unconfigured_dict)
        self.state_configured.register(configured_dict)
        self.state_running.register(running_dict)
        self.state_enabled.register(enabled_dict)

        # Start with a default state.
        self._state = self.state_unconfigured

    def on_transition(self, transition):
        # The next state will be the result of the on_transition function.
        self._state = self._state.on_transition(transition)

    def state(self):
        return self._state

    def configfunc(self):
        print ("Running configfunc()")
        return True

    def unconfigfunc(self):
        print ("Running unconfigfunc()")
        return True

    def beginrunfunc(self):
        print ("Running beginrunfunc()")
        return True

    def endrunfunc(self):
        print ("Running endrunfunc()")
        return True

    def enablefunc(self):
        print ("Running enablefunc()")
        return True

    def disablefunc(self):
        print ("Running disablefunc()")
        return True


def test():

    yy = TestStateMachine()
    print("TestStateMachine state:", yy.state())
    assert(yy.state() == yy.state_unconfigured)

    yy.on_transition(yy.transition_configure)
    print("TestStateMachine state:", yy.state())
    assert(yy.state() == yy.state_configured)

    yy.on_transition(yy.transition_beginrun)
    print("TestStateMachine state:", yy.state())
    assert(yy.state() == yy.state_running)

    yy.on_transition(yy.transition_enable)
    print("TestStateMachine state:", yy.state())
    assert(yy.state() == yy.state_enabled)

    yy.on_transition(yy.transition_disable)
    print("TestStateMachine state:", yy.state())
    assert(yy.state() == yy.state_running)

    yy.on_transition(yy.transition_endrun)
    print("TestStateMachine state:", yy.state())
    assert(yy.state() == yy.state_configured)

    yy.on_transition(yy.transition_unconfigure)
    print("TestStateMachine state:", yy.state())
    assert(yy.state() == yy.state_unconfigured)

    return

if __name__ == '__main__':
    test()
