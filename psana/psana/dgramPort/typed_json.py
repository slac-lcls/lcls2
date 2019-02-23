import numpy as np
import numbers
import re

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
# This file has three external APIs:
#      validate_typed_json(d) checks if the above rules are followed for dictionary d, and returns True/False.
#
#      write_typed_json(filename, d) writes the dictionary d as JSON to the given filename.
#
#      class cdict is a helper class for building typed JSON dictionaries.
#          cdict(old_cdict) - The constructor optionally takes an old cdict to clone.
#          set(name, value, type="INT32", override=False, append=False)
#                           - This routine is the heart of the class.  It traverses the name hierarchy
#                             to set a new value.  The value maybe a numeric value, a list of numeric
#                             values, a numpy array, a clist, or a list of clists.  For numeric values or
#                             lists of numeric values, type indicates the desired type of a new attribute.
#                             For a clist or list of clists, append indicates if the target should be
#                             overwritten or the new clists appended to an existing list.  In general,
#                             once a name is typed, setting a new value will not change the type unless
#                             the override flag is True.  This returns True if the set was successful
#                             and false otherwise.  The contents of a clist are copied, but any ndarrays
#                             are not.  (Therefore, modifying a clist after assigning it to a name does
#                             not change the assigned value, but modifying the contents of an ndarray does.)
#
#          setInfo(detType=None, detName=None, detId=None, doc=None)
#                           - Sets the additional information for a top-level clist.
#          setAlg(alg, version=[0,0,0], doc="")
#                           - Set the algorithm information.
#          writeFile(filename)
#                           - Write the typed JSON to a file.
#

typerange = {
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

typedict = {
    "UINT8"  : "uint8", 
    "UINT16" : "uint16", 
    "UINT32" : "uint32", 
    "UINT64" : "uint64", 
    "INT8"   : "int8", 
    "INT16"  : "int16", 
    "INT32"  : "int32", 
    "INT64"  : "int64", 
    "FLOAT"  : "float32", 
    "DOUBLE" : "float64", 
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
            if not v[0] in typerange.keys():
                return namify(top + [n]) + " has invalid type " + str(v[0])
            vv = typerange[v[0]]
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
            vv = typerange[v[0]]
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

def write_typed_json(filename, d):
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
        return True

#
# Let's try to make creating valid dictionaries easier.  This heart
# of this class is the method:
#     set(name, value, type="INT32", override=False, append=False)
# Once the type of a name is set, changing it is only possible if
# override is True.
#
# name here is an expanded name (with "_") that will be unpacked to
# build the hierarchy.  Multiple possibilities for value are supported:
#     - A numeric value or numpy array will just create/overwrite the 
#       value.
#     - A list of numeric values will be converted to a numpy array
#       of the specified type and stored.
#     - A cdict will splice in the hierarchy at the specified location.
#     - A list of cdicts will add the list of dictionaries.
#
class cdict(object):
    def __init__(self, old=None):
        self.dict = {}
        if isinstance(old, cdict):
            self.dict.update(old.dict)

    def splitname(self, name):
        n = name.split("_")
        r = []
        for nn in n:
            m = re.search('^(.*[^0-9])([0-9]+)$', nn)
            if m is None:
                if not re.search('^[^0-9]', nn):
                    return None
                r.append(nn)
            else:
                r.append(m.group(1))
                r.append(int(m.group(2)))
        return r

    def get(self, name):
        if len(name) == 0:
            return None
        n = self.splitname(name)
        if n is None:
            return None
        d = self.dict
        while len(n) != 0:
            if isinstance(n[0], int):
                if isinstance(d, list):
                    try:
                        d = d[n[0]]
                        n = n[1:]
                    except:
                        return None
                else:
                    return None
            else:
                if isinstance(d, dict):
                    try:
                        d = d[n[0]]
                        n = n[1:]
                    except:
                        return None
                else:
                    return None
        if isinstance(d, tuple):
            return d[1]
        else:
            return d

    def checknumlist(self, l):
        for v in l:
            if isinstance(v, list):
                if not self.checknumlist(v):
                    return False
            elif not isinstance(v, numbers.Number):
                return False
        return True

    def set(self, name, value, type="INT32", override=False, append=False):
        if len(name) == 0:
            return False
        n = self.splitname(name)
        if n is None:
            return False
        d = self.dict
        # Check the type of value!
        if isinstance(value, numbers.Number):
            if not type in typedict.keys():
                return False
            value = (type, value)
            issimple = True
        elif isinstance(value, np.ndarray):
            issimple = True
        elif isinstance(value, cdict):
            issimple = False
        elif isinstance(value, list):
            if self.checknumlist(value):
                value = np.array(value, dtype=type.lower())
                issimple = True
            else:
                # Must be a list of cdicts!
                for v in value:
                    if not isinstance(v, cdict):
                        return False
                issimple = False
        else:
            return False
        for i in range(len(n)):
            if isinstance(n[i], int):
                if d is None:
                    p[n[i-1]] = []
                    d = p[n[i-1]]
                if not isinstance(d, list):
                    if override:
                        p[n[i-1]] = []
                        d = p[n[i-1]]
                    else:
                        return False
                while len(d) < n[i]+1:
                    d.append({})
                p = d
                d = d[n[i]]
            else:
                if d is None:
                    p[n[i-1]] = {}
                    d = p[n[i-1]]
                if not isinstance(d, dict):
                    if override:
                        p[n[i-1]] = {}
                        d = p[n[i-1]]
                    else:
                        return False
                try:
                    p = d
                    d = d[n[i]]
                except:
                    if i+1 == len(n):
                        if not issimple and append:
                            p[n[i]] = []
                        else:
                            p[n[i]] = None
                    elif isinstance(n[i+1], int):
                        p[n[i]] = []
                    else:
                        p[n[i]] = {}
                    d = p[n[i]]
        # d is current value, if any. p is the last enclosing structure.
        if issimple:
            # A number or a numpy array
            if isinstance(value, np.ndarray):
                if not override and d is not None:
                    if not isinstance(d, np.ndarray):
                        return False
                    if value.dtype != d.dtype or value.shape != d.shape:
                        return False
            else:
                if not override and d is not None:
                    if d[0] != "DOUBLE" and d[0] != "FLOAT":
                        value = (d[0], int(value[1]))
                    else:
                        value = (d[0], value[1])
            p[n[-1]] = value
            return True
        else:
            # A cdict or a list of cdicts
            if isinstance(value, cdict):
                if not override and d is not None and not (isinstance(d, dict) or 
                                                           (isinstance(d, list) and append)):
                    return False
                vnew = {}
                vnew.update(value.dict)
                if isinstance(d, list):
                    d.append(vnew)
                else:
                    p[n[-1]] = vnew
            else:
                if not override and d is not None and not isinstance(d, list):
                    return False
                if not append:
                    p[n[-1]] = []
                for v in value:
                    vnew = {}
                    vnew.update(v)
                    p[n[-1]].append(vnew)
            return True

    def setAlg(self, alg, version=[0,0,0], doc=""):
        if not isinstance(version, list) or len(version) != 3:
            print("version should be a length 3 list!\n")
            return
        if not isinstance(alg, str):
            print("alg should be a string!")
            return
        if not isinstance(doc, str):
            print("doc should be a string!")
            return
        self.dict["alg"] = {
            "alg": alg,
            "doc": doc,
            "version": version
        }

    def setString(self, name, value):
        if value is not None:
            if not isinstance(value, str):
                print("%s must be a str!" % name)
            else:
                self.dict[name] = value

    def setInfo(self, detType=None, detName=None, detId=None, doc=None):
        self.setString("detType", detType)
        self.setString("detName", detName)
        self.setString("detId", detId)
        self.setString("doc", doc)

    def writeFile(self, file):
        return write_typed_json(file, self.dict)
