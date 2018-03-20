from ami import operation

class GraphConfigError(Exception):
    pass

class GraphRuntimeError(Exception):
    pass

class Graph(object):
    def __init__(self, store, graph_cfg=None):
        self.cfg_old = {}
        if graph_cfg is None:
            self.cfg = {}
        else:
            self.cfg = graph_cfg
        self.subs = {}
        self.store = store
        self.operations = {}

    def subscribe(self, result, opid):
        if result in self.subs:
            self.subs[result].append(opid)
        else:
            self.subs[result] = [opid]


    def configure(self):
        self.subs = {}
        self.operations = {}
        for name, op_cfg in self.cfg.items():
            if name in self.operations:
                raise GraphConfigError("cannot add operation %s to graph - name already in use"%name)
            else:
                try:
                    cfg_obj = getattr(operation,  op_cfg['optype'])(**op_cfg['config'])
                    if not cfg_obj.valid:
                        raise GraphConfigError("operation %s has an invalid configuration"%name)
                    self.operations[name] = operation.Operation(name, self.store, op_cfg['inputs'], cfg_obj)
                    for inp in self.operations[name].inputs:
                        self.subscribe(inp['name'], name)
                except Exception as exp:
                    raise GraphConfigError("exception encounterd adding operation %s to the graph: %s"%(name, exp))

    def revert(self):
        self.cfg = self.cfg_old
        self.configure()

    def execute(self, srcs):
        pending_ops = []
        for src in srcs:
            if self.store.is_updated(src) and src in self.subs:
                pending_ops += self.subs[src]
        index = 0
        while index < len(pending_ops):
            try:
                self.operations[pending_ops[index]].run()
                if self.store.is_updated(pending_ops[index]) and pending_ops[index] in self.subs:
                    pending_ops += self.subs[pending_ops[index]]
            except Exception as exp:
                raise GraphRuntimeError("exception encounterd running operation %s on the graph: %s"%(pending_ops[index], exp))
            index += 1 
        
    def update(self, new_graph_cfg):
        self.cfg_old = self.cfg
        self.cfg = new_graph_cfg
