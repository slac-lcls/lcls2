import numpy as np
import numbers

#
# The goal here is to assist the writing of JSON files from python.  The assumption here is that
# we pass in a dictionary, the keys being the names, and the values being one of the following:
#     - A length two tuple, representing a scalar: ("TYPESTRING", VALUE)
#     - A dictionary, representing a hierarchical name space.
#     - A numpy array.
#     - A list of dictionaries.
#
# Furthermore, the dictionary must contain a few additional keys with specific value types:
#     - "detType" maps to a str.
#     - "detName" maps to a str.
#     - "detId" maps to a str.
#     - "doc" is optional, but if present maps to a str.
#     - "alg" is optional, but if present maps to a dictionary.  The keys of this dictionary are:
#         - "alg" which maps to a str.
#         - "doc" which is optional, but if present maps to a str.
#         - "version" which maps to a list of three integers.
#

typedict = {
    "UINT8"  : (0, 2**8 - 1), 
    "UINT16" : (0, 2**16 - 1),
    "UINT32" : (0, 2**32 - 1),
    "UINT64" : (0, 2**64 - 1),
    "INT8"   : (-2**7, 2**7 - 1), 
    "INT16"  : (-2**15, 2**15 - 1), 
    "INT32"  : (-2**31, 2**31 - 1), 
    "INT64"  : (-2**63, 2**63 - 1), 
    "FLOAT"  : None,
    "DOUBLE" : None
}

nptypedict = {
    np.dtype("uint8")  : ("UINT8",  False),
    np.dtype("uint16") : ("UINT16", False),
    np.dtype("uint32") : ("UINT32", False),
    np.dtype("uint64") : ("UINT64", False),
    np.dtype("int8")   : ("INT8",   False),
    np.dtype("int16")  : ("INT16",  False),
    np.dtype("int32")  : ("INT32",  False),
    np.dtype("int64")  : ("INT64",  False),
    np.dtype("float32"): ("FLOAT",  True),
    np.dtype("float64"): ("DOUBLE", True)
}

def namify(l):
    s = ""
    for v in l:
        if isinstance(v, str):
            if s == "":
                s = v
            else:
                s = s + "_" + v
        else:
            s = s + str(v)
    return s

#
# Return None for a valid dictionary and an error string otherwise.
#
def validate_typed_json(d, top=[]):
    if not isinstance(d, dict):
        return "Not a dictionary"
    k = list(d.keys())
    if len(top) == 0:
        for f in ["detType", "detName", "detId"]:
            if not f in k or not isinstance(d[f], str):
                return "No valid " + f
            else:
                k.remove(f)
        if "doc" in k and not isinstance(d["doc"], str):
            return "No valid doc"
        else:
            k.remove("doc")
        if "alg" in k:
            a = d["alg"]
            if not isinstance(a, dict):
                return "alg is not a dictionary"
            ak = a.keys()
            if not "alg" in ak or not isinstance(a["alg"], str):
                return "alg has no valid alg"
            if "doc" in ak and not isinstance(a["doc"], str):
                return "alg has no valid doc"
            if not "version" in ak or not isinstance(a["version"], list) or len(a["version"]) != 3:
                return "alg has no valid version"
            # Do we want to check if the three elements are integers?!?
            k.remove("alg")
    for n in k:
        v = d[n]
        if isinstance(v, dict):
            r = validate_typed_json(v, top + [n])
            if r is not None:
                return r
        elif isinstance(v, list):
            for (nn, dd) in enumerate(v):
                if not isinstance(dd, dict):
                    return namify(top + [n, nn]) + " is not a dict"
                r = validate_typed_json(dd, top + [str(nn)])
                if r is not None:
                    return r
        elif isinstance(v, tuple):
            if len(v) != 2:
                return namify(top + [n]) + " should have len 2"
            if not v[0] in typedict.keys():
                return namify(top + [n]) + " has invalid type " + str(v[0])
            vv = typedict[v[0]]
            if vv is None:
                if not isinstance(v[1], numbers.Number):
                    return namify(top + [n]) + " value is not a number"
            else:
                if not isinstance(v[1], int):
                    return namify(top + [n]) + n + " value is not an int"
                if v[1] < vv[0] or v[1] > vv[1]:
                    return namify(top + [n]) + n + " is out of range"
        elif isinstance(v, np.ndarray):
            pass
        else:
            return namify(top + [n]) + " is invalid"

def write_json_dict(f, d, tdict, top=[], indent="    "):
    prefix = indent
    for n in d.keys():
        v = d[n]
        if isinstance(v, str):
            f.write('%s"%s": "%s"' % (prefix, n, v))
        if isinstance(v, dict):
            if n == "alg":
                try:
                    doc = v["doc"]
                except:
                    doc = ""
                vinfo = v["version"]
                f.write('%s"alg": {"alg": "%s", "doc": "%s", "version": [%d, %d, %d] }' %
                        (prefix, v["alg"], v["doc"], vinfo[0], vinfo[1], vinfo[2]))
            else:
                f.write('%s"%s": {\n' % (prefix, n))
                tdict0 = {}
                write_json_dict(f, v, tdict0, top + [n], indent + "    ")
                f.write('\n%s}' % indent)
                tdict[n] = tdict0
        elif isinstance(v, list):
            if isinstance(v[0], dict):
                f.write('%s"%s": [\n' % (prefix, n))
                tdict0 = {}
                for (nn, dd) in enumerate(v):
                    if nn != 0:
                        f.write(',\n')
                    f.write('%s    {\n' % indent)
                    write_json_dict(f, dd, tdict0, top + [n, nn], indent + "        ")
                    f.write('\n%s    }' % indent)
                f.write('\n%s]' % indent)
                tdict[n] = tdict0
            else:
                # We must be writing type information
                f.write('%s"%s": ["%s"' % (prefix, n, v[0]))
                for nn in range(1, len(v)):
                    f.write(', %d' % v[nn])
                f.write(']')
        elif isinstance(v, tuple):
            vv = typedict[v[0]]
            if vv is None:
                f.write('%s"%s": %g' % (prefix, n, v[1]))
            else:
                f.write('%s"%s": %d' % (prefix, n, v[1]))
            tdict[n] = v[0]
        elif isinstance(v, np.ndarray):
            typ = nptypedict[v.dtype]
            f.write('%s"%s": [' % (prefix, n))
            start = ""
            for vv in v.ravel():
                if typ[1]:
                    f.write('%s%g' % (start, vv))
                else:
                    f.write('%s%d' % (start, vv))
                start = ", "
            f.write(']')
            tdict[n] = list((typ[0],) + v.shape)
        prefix = ",\n" + indent

def make_typed_json(filename, d):
    r = validate_typed_json(d)
    if r is not None:
        print(r)
        return False
    with open(filename, "w") as f:
        f.write('{\n')
        tdict = {}
        write_json_dict(f, d, tdict)
        f.write(',\n    "json_types": {\n')
        write_json_dict(f, tdict, {}, [], "        ")
        f.write('\n    }\n}\n')
