"""
Basic Operations Module
"""
import json
import importlib

class OperationError(Exception):
    def __init__(self, opid, message):
        self.opid = opid
        self.message = message

    def __repr__(self):
        return "%s(%s, %s)"%(self.__class__.__name__, self.opid, self.message)

    def __str__(self):
        return "%s (source opid %s)"%(self.args[0], self.message)


class NodeConfig(object):
    def __init__(self, name, opname, *inputs):
        self.name = name
        self.opname = opname
        self.inputs = []
        for inp in inputs:
            self.inputs.append({"name": inp, "required": True})
        self.config = {}

    def add_to_config(self, key, value):
        self.config[key] = value

    def export(self):
        cfg = {
            "optype": self.opname,
            "inputs": self.inputs,
            "config": self.config,
        }
        return cfg


class OpConfig(object):
    def __init__(self, *keys):
        self.uses_kwargs = False
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


class Eval(OpConfig):
    def __init__(self, expression, imports=None):
        super(__class__, self).__init__("expression")
        self.uses_kwargs = True
        self.imports = {}
        if imports is not None:
            for imp in imports:
                self.imports[imp] = importlib.import_module(imp)
        self.expression = compile(expression, '<string>', 'eval')

    def operate(self, **kargs):
        return eval(self.expression, self.imports, kargs)


class EvalNode(NodeConfig):
    def __init__(self, name, expression, *inputs):
        super(__class__, self).__init__(name, "Eval", *inputs)
        self.add_to_config("expression", expression)

    def imports(self, *imports):
        self.add_to_config("imports", imports)


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
            if self.config.uses_kwargs:
                input_data = {}
                for inp in self.inputs:
                    input_data[inp["name"]] = self.store.get(inp["name"])
                self.store.put(self.name, self.config.operate(**input_data))
            else:
                input_data = []
                for inp in self.inputs:
                    input_data.append(self.store.get(inp["name"]))
                self.store.put(self.name, self.config.operate(*input_data))
