import ami.operation

class Graph(object):
    def __init__(self, store, graph_cfg=None):
        if graph_cfg is None:
            self.cfg = {}
        else:
            self.cfg = graph_cfg
        self.subs = {}
        self.store = store
        self.operations = {}

    def configure(self):
        for opid, data in self.store.items():
            print(opid)

    def execute(self):
        pass

    def update(self, new_graph_cfg):
        self.cfg = new_graph_cfg
