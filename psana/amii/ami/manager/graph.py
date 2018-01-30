class Graph():

    def __init__(self,graph_dict):
        self.graph_dict = graph_dict
        pass

    def add(self, box, name, parents):
        # parents = list of 2-tuple (parent_node, input_key_name)

        # update the outputs of parent boxes
        for p in parents:
            pnode, input_key_name = p
            self.graph_dict[pnode]['config']['outputs'].append((name, input_key_name))
        # add ourselves to the graph
        self.graph_dict[name] = box
