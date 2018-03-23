"""
Basic Operations Module
"""
import json

class OperationError(Exception):
    def __init__(self, opid, message):
        self.opid = opid
        self.message = message

    def __repr__(self):
        return "%s(%s, %s)"%(self.__class__.__name__, self.opid, self.message)

    def __str__(self):
        return "%s (source opid %s)"%(self.args[0], self.message)


class OpConfig(object):
    def __init__(self, *keys):
        self.required = set(keys) # the list of required parameters

    @property
    def valid(self):
        for key in self.required:
            if not hasattr(self, key):
                return False
        return True

    def export(self):
        cfg = {}
        for key in self.required:
            cfg[key] = getattr(self, key)
        return cfg


class Operation(object):
    def __init__(self, name, store, inputs, config):
        self.name = name
        self.store = store
        self.inputs = inputs
        self.config = config

    def inputs_ready(self):
        for data_input in self.inputs:
           if data_input["required"] and not self.store.is_updated(data_input["name"]):
                return False 
        return True

    def run(self):
        if self.inputs_ready():
            input_data = []
            for inp in self.inputs:
                input_data.append(self.store.get(inp["name"]))
            self.store.put(self.name, self.config.operate(*input_data))
