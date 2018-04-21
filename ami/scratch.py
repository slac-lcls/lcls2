
import json
import numpy as np



def simple_tree(json):
    """
    Generate a dependency tree from a json input

    Parameters
    ----------
    json : json object

    Returns
    -------
    graph : dict
        dict where the keys are operations, the value are upstream
        (dependent) operations that should be done first
    """
    graph = {}
    for op in json:
        graph[op] = json[op]['inputs']
    return graph


def resolve_dependencies(dependencies):
    """
    Dependency resolver

    Parameters
    ----------
    dependencies : dict
        the values are the dependencies of their respective keys

    Returns
    -------
    order : list
        a list of the order in which to resolve dependencies
    """

    # this just makes sure there are no duplicate dependencies
    d = dict( (k, set(dependencies[k])) for k in dependencies )

    # output list
    r = []

    while d:

        # find all nodes without dependencies (roots)
        t = set(i for v in d.values() for i in v) - set(d.keys())

        # and items with no dependencies (disjoint nodes)
        t.update(k for k, v in d.items() if not v)

        # both of these can be resolved
        r.extend(t)

        # remove resolved depedencies from the tree
        d = dict(((k, v-t) for k, v in d.items() if v))

    return r


class FeatureStore(object):

    def __init__(self):
        self.namespace = {'cspad' : np.random.randn(1000,1000)}
        return

    def put(self, name, value):
        print('PUT:' + name)
        self.namespace[name] = value


j = json.load(open('examples/python.json'))
t = simple_tree(j)
ops = resolve_dependencies(t)

st = FeatureStore()


for op in ops:
    if op in st.namespace:
        continue
    store_put = '\n' + '; '.join(['store.put("%s", %s)'%(output, output) for output in j[op]['outputs']]) + '\n'
    loc = {'store': st}
    loc.update(st.namespace)
    exec(j[op]['code'] + store_put, {"np" : np}, loc)


