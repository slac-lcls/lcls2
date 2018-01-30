"""
Basic Operations Module
"""
import six
import json

class OperationError(Exception):
    def __init__(self, opid, message):
        self.opid = opid
        self.message = message

    def __repr__(self):
        return "%s(%s, %s)"%(self.__class__.__name__, self.opid, self.message)

    def __str__(self):
        return "%s (source opid %s)"%(self.args[0], self.message)


class ConfigurationError(OperationError):
    pass


class Datagram(object):
    def __init__(self, event_id, config_id, data=None):
        self.event_id = event_id
        self.config_id = config_id
        self.data = data

    def __str__(self):
        return "Datagram:\n event_id: %s\n config_id: %s\n data: %s"%(self.event_id, self.config_id, self.data)

class OpConfig(dict):
    def __init__(self):
        self.required_config = set(['outputs']) # the list of required parameters

    def require(self, *keys):
        for key in keys:
            self.required_config.add(key)

    @property
    def valid(self):
        for key in self.required_config:
            if key not in self:
                return False
        return True

    #def write_json(self):
    #    json.dump(self.required_config)

class Operation(object):
    def __init__(self, opid, ops):
        self.opid = opid
        self.event_id = -1
        self.config_id = None
        self.ops = ops
        self.config = OpConfig()
        # keeps track of which inputs have been modified, like an event-builder
        self.inputs = {}
        self.outputs = {}
        self.configured = False

    def add_input(self, inp):
        self.inputs[inp] = self.event_id

    def _check_inputs(self):
        for val in six.itervalues(self.inputs):
            if val != self.event_id:
                return False
        return True

    def configure(self):
        # the user can implement a specialized configure method to do special stuff
        return True

    def run(self):
        return True

    def _configure(self, config_id, global_config):
        if config_id != self.config_id: 
            self.outputs.clear()
            self.config_id = config_id
            self.configured = False
            # get our specific config from the global config
            self.config.update(global_config[self.opid]['config'])
            if not self.config.valid:
                raise ConfigurationError(self.opid, "Missing required configuration key")
            self.configured = self.configure()
            if not self.configured:
                raise ConfigurationError(self.opid, "User configure function returned a failed state")
            for operation_id, _, _ in self.config['outputs']:
                self.ops[operation_id]._configure(config_id, global_config)

    def _run(self, inp, indata):
        if not self.configured:
            raise OperationError(self.opid, "Operation must be in a configured state before calling run!")
        if not inp in self.inputs:
            raise OperationError(self.opid, "Operation does not have input: '%s", inp)
        if (indata is not None) and (indata.config_id == self.config_id) and (indata.event_id > self.inputs[inp]):
            if indata.event_id > self.event_id:
                # new event seen
                self.event_id = indata.event_id
            self.inputs[inp] = indata.event_id
            setattr(self, inp, indata.data)
            if self._check_inputs():
                if self.run():
                    for operation_id, input_name, output_name  in self.config['outputs']:
                        if output_name in self.outputs:
                            self.ops[operation_id]._run(input_name, Datagram(self.event_id, self.config_id, self.outputs[output_name]))

# we think OpConfig should be a dictionary with a fixed set of keys, per operation (e.g. ROI):
# - required_config (operation parameters)
# - outputs
# - optype (__class__.__name__)
# at the moment, json is useful for serializing to disk, but zmq could use pickle: "json" may be a detail
# it's important for each window to know "who we are" in the graph so it can edit the graph

# the "dict" here is the json
