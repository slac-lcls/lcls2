
import json

def simple_tree(json):
    graph = {}
    for op in json:
        graph[op] = json[op]['inputs']
    return graph


def build_dep_tree(json):

    graph = {}

    for op in json:
        for inps in json[op]['inputs']:
            if inps not in graph:
                graph[inps] = set()
            graph[inps].add(op)

    return graph


def dep(arg):
    '''
        Dependency resolver

    "arg" is a dependency dictionary in which
    the values are the dependencies of their respective keys.
    '''
    d=dict((k, set(arg[k])) for k in arg)
    r=[]
    while d:
        # values not in keys (items without dep)
        t=set(i for v in d.values() for i in v)-set(d.keys())
        # and keys without value (items without dep)
        t.update(k for k, v in d.items() if not v)
        # can be done right away
        r.append(t)
        # and cleaned up
        d=dict(((k, v-t) for k, v in d.items() if v))
    return r


j = json.load(open('examples/python.json'))


#t = build_dep_tree(j)
t = simple_tree(j)
print dep(t)

