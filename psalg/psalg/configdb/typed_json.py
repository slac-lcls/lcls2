import numpy as np
import numbers
import re

#
# The goal here is to assist the writing of JSON files from python.  The 
# assumption here is that we pass in a dictionary, the keys being the names,
# and the values being one of the following:
#     - A length two tuple, representing a scalar: ("TYPESTRING", VALUE)
#     - A dictionary, representing a hierarchical name space.
#     - A numpy array.
#     - A list of dictionaries.
#
# Furthermore, the dictionary must contain a few additional keys with specific
# value types:
#     - "detType" maps to a str.
#     - "detName" maps to a str.
#     - "detId" maps to a str.
#     - "doc" is optional, but if present maps to a str.
#     - "alg" is optional, but if present maps to a dictionary.  The keys of 
#       this dictionary are:
#         - "alg" which maps to a str.
#         - "doc" which is optional, but if present maps to a str.
#         - "version" which maps to a list of three integers.
#
# This file has several external APIs:
#      validate_typed_json(d, edef={}) checks if the above rules are followed
#      for dictionary d, with a dictionary of enum definitions edef and returns
#      True/False.
#
#      write_typed_json(filename_or_fd, d) writes the dictionary d as JSON 
#      to the given filename (or file descriptor).
#
#      class cdict is a helper class for building typed JSON dictionaries.
#          cdict(old_cdict) - The constructor optionally takes an old cdict to clone.
#          set(name, value, type="INT32", override=False, append=False)
#                           - This routine is the heart of the class.  It 
#                             traverses the name hierarchy to set a new value.
#                             The value maybe a numeric value, a list of numeric
#                             values, a numpy array, a clist, or a list of 
#                             clists.  For numeric values or lists of numeric 
#                             values, type indicates the desired type of a new
#                             attribute.  For a clist or list of clists, append
#                             indicates if the target should be overwritten or
#                             the new clists appended to an existing list.  In
#                             general, once a name is typed, setting a new value
#                             will not change the type unless the override flag
#                             is True.  This returns True if the set was
#                             successful and false otherwise.  The contents of
#                             a clist are copied, but any ndarrays are not.  
#                             (Therefore, modifying a clist after assigning it
#                             to a name does not change the assigned value, but
#                             modifying the contents of an ndarray does.)
#
#          setInfo(detType=None, detName=None, detId=None, doc=None)
#                           - Sets the additional information for a top-level
#                             clist.
#          setAlg(alg, version=[0,0,0], doc="")
#                           - Set the algorithm information.
#          writeFile(filename)
#                           - Write the typed JSON to a file.
#
# The following routines work on typed JSON dictionaries, or a dictionary
# that maps strings to typed JSON dictionaries, such as those returned from
# configdb.
#      getValue(dict, name)
#          - Given a dictionary, retrieve the value of the fully-dotted name.
#      getType(dict, name)
#          - Given a dictionary, retrieve the type of the fully-dotted name.
#            This will be either:
#                - A string representing a basic type.
#                - A dictionary representing an enum type.
#                - A list, the first element of which is one of the two above
#                  elements and the remainder of which is integer array
#                  dimensions.
#      updateValue(dict, name, string_value)
#          - Given a dictionary, set the element specified by name to the
#            value in string_value.  Array values will be space-separated.
#            This routine returns an integer:
#                0 if successful.
#                1 if the path does not exist.
#                2 if the type conversion failed
#                3 if typed JSON dictionary is somehow malformed.

typerange = {
    "UINT8"   : (0, 2**8 - 1), 
    "UINT16"  : (0, 2**16 - 1),
    "UINT32"  : (0, 2**32 - 1),
    "UINT64"  : (0, 2**64 - 1),
    "INT8"    : (-2**7, 2**7 - 1), 
    "INT16"   : (-2**15, 2**15 - 1), 
    "INT32"   : (-2**31, 2**31 - 1), 
    "INT64"   : (-2**63, 2**63 - 1), 
    "FLOAT"   : None,
    "DOUBLE"  : None,
    "CHARSTR" : None,
}

typedict = {
    "UINT8"   : "uint8", 
    "UINT16"  : "uint16", 
    "UINT32"  : "uint32", 
    "UINT64"  : "uint64", 
    "INT8"    : "int8", 
    "INT16"   : "int16", 
    "INT32"   : "int32", 
    "INT64"   : "int64", 
    "FLOAT"   : "float32", 
    "DOUBLE"  : "float64",
    "CHARSTR" : "str",
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
                s = s + "." + v
        else:
            s = s + "." + str(v)
    return s

def splitname(name):
    n = name.split(".")
    r = []
    for nn in n:
        if nn[0].isdigit():
            try:
                r.append(int(nn))
            except:
                return None
        else:
            r.append(nn)
    return r

def pythonizeName(name):
    n = splitname(name)
    r = n[0]
    for i in n[1:]:
        if isinstance(i, int):
            r += "_" + str(i)
        else:
            r += "." + i
    return r

#
# Return None for a valid dictionary and an error string otherwise.
#
def validate_typed_json(d, edef={}, top=[], headers=True):
    if not isinstance(d, dict):
        return "Not a dictionary"
    k = list(d.keys())
    if len(top) == 0:
        for f in ["detType", "detName", "detId"]:
            if headers and (not f in k or not isinstance(d[f], str)):
                return "No valid " + f
            if f in k:
                k.remove(f)
        if "doc" in k:
            if headers and not isinstance(d["doc"], str):
                return "No valid doc"
            k.remove("doc")
        if "alg" in k:
            if headers:
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
            r = validate_typed_json(v, edef, top + [n])
            if r is not None:
                return r
        elif isinstance(v, list):
            for (nn, dd) in enumerate(v):
                if not isinstance(dd, dict):
                    return namify(top + [n, nn]) + " is not a dict"
                r = validate_typed_json(dd, edef, top + [str(nn)])
                if r is not None:
                    return r
        elif isinstance(v, tuple):
            if len(v) != 2:
                return namify(top + [n]) + " should have len 2"
            if v[0] in edef.keys() and (isinstance(v[1], numbers.Number) or isinstance(v[1], np.ndarray)):
                return None
            if not v[0] in typerange.keys() and not v[0] in edef.keys():
                return namify(top + [n]) + " has invalid type " + str(v[0])
            vv = typerange[v[0]]
            if vv is None:
                if v[0] != "CHARSTR" and not isinstance(v[1], numbers.Number):
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
        return None

def write_json_dict(f, d, edef, tdict, top=[], indent="    ", **hw):
    prefix = indent
    k = list(d.keys())
    try:
        if not hw['headers']:
            for ff in ["detType", "detName", "detId", "doc", "alg"]:
                try: 
                    k.remove(ff)
                except:
                    pass
    except:
        pass
    for n in k:
        v = d[n]
        if isinstance(v, str):
            f.write('%s"%s": "%s"' % (prefix, n, v))
        elif isinstance(v, dict):
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
                write_json_dict(f, v, edef, tdict0, top + [n], indent + "    ")
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
                    write_json_dict(f, dd, edef, tdict0, top + [n, nn], indent + "        ")
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
            if v[0] in edef.keys():
                if isinstance(v[1], numbers.Number):
                    f.write('%s"%s": %d' % (prefix, n, v[1]))
                    tdict[n] = v[0]
                elif isinstance(v[1], np.ndarray):
                    f.write('%s"%s": [' % (prefix, n))
                    start = ""
                    for vv in v[1].ravel():
                        f.write('%s%d' % (start, vv))
                        start = ", "
                    f.write(']')
                    tdict[n] = list((v[0],) + v[1].shape)
            else:
                vv = typerange[v[0]]
                if vv is None:
                    if v[0] == "CHARSTR":
                        f.write('%s"%s": "%s"' % (prefix, n, v[1]))
                    else:
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
        else: # Must be an enum!
            f.write('%s"%s": %d' % (prefix, n, v))
        prefix = ",\n" + indent

def write_typed_json(filename_or_fd, d, edef, headers=True):
    r = validate_typed_json(d, edef, [], headers)
    if r is not None:
        print(r)
        return False
    if isinstance(filename_or_fd, str):
        f = open(filename_or_fd, "w")
    else:
        f = filename_or_fd
    f.write('{\n')
    tdict = {}
    write_json_dict(f, d, edef, tdict, headers=headers)
    f.write(',\n    ":types:": {\n')
    if edef != {}:
        f.write('        ":enum:": {\n')
        write_json_dict(f, edef, {}, {}, [], "            ")
        f.write('\n        },\n')
    f.write('        "detType": "CHARSTR",\n')
    f.write('        "detName": "CHARSTR",\n')
    f.write('        "detId": "CHARSTR",\n')
    f.write('        "doc": "CHARSTR",\n')
    f.write('        "alg": {\n')
    f.write('            "alg": "CHARSTR",\n')
    f.write('            "doc": "CHARSTR",\n')
    f.write('            "version": ["INT32", 3]\n')
    f.write('        },\n')
    write_json_dict(f, tdict, {}, {}, [], "        ")
    f.write('\n    }\n}\n')
    return True

#
# Let's try to make creating valid dictionaries easier.  This heart
# of this class is the method:
#     set(name, value, type="INT32", override=False, append=False)
# Once the type of a name is set, changing it is only possible if
# override is True.
#
# name here is an expanded name (with ".") that will be unpacked to
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
        self.enumdef = {}
        if isinstance(old, cdict):
            self.dict.update(old.dict)
            self.enumdef.update(old.enumdef)
        elif isinstance(old, dict) and ":types:" in old.keys():
            cp = {}
            cp.update(old)
            jt = cp[':types:']
            if ":enum:" in jt.keys():
                self.enumdef = jt[":enum:"]
                del jt[":enum:"]
            del cp[':types:']
            self.init_from_json(cp, jt)

    def init_from_json(self, old, jt, base=None):
        if isinstance(old, dict):
            for k in old.keys():
                if k in ["detType", "detName", "detId", "doc"]:
                    self.dict[k] = str(old[k])
                    continue
                if k == "alg":
                    alg = old['alg']
                    if isinstance(alg, dict) and set([k for k in alg.keys()]) == set(["alg", "doc", "version"]):
                        self.dict['alg'] = alg
                    continue
                v = old[k]
                t = jt[k]
                if base is None:
                    n = k
                else:
                    n = base + "." + k
                if isinstance(v, dict) or (isinstance(v, list) and not self.checknumlist(v)):
                    self.init_from_json(v, t, n)
                elif isinstance(v, list) or isinstance(v, np.ndarray):
                    if isinstance(v, list):
                        # set v to an np.array of the appropriate type!
                        v = np.array(v, dtype=typedict[t[0]]).reshape(t[1:])
                        pass
                    self.set(n, v)
                else:
                    # Scalar!
                    self.set(n, v, t)
        elif isinstance(old, list):
            for (k, v) in enumerate(old):
                n = base + str(k)
                self.init_from_json(v, jt, n)

    def typed_json(self):
        (d, t) = self.create_json(self.dict, True)
        if self.enumdef != {}:
            t[":enum:"] = {}
            t[":enum:"].update(self.enumdef)
        d[':types:'] = t
        return d

    # Add all of t2 to t1.  We'd use update, but we want to merge all 
    # the way down...
    def merge_dict(self, t1, t2):
        for k in t2.keys():
            if k in t1.keys():
                if isinstance(t1[k], dict):
                    if isinstance(t2[k], dict):
                        self.merge_dict(t1[k], t2[k])
                    # else WTF?
                # else check if types agree?!?
            else:
                t1[k] = t2[k]

    def create_json(self, input, top=False):
        if isinstance(input, dict):
            d = {}
            t = {}
            for k in input.keys():
                if top and (k in ["detType", "detName", "detId", "doc", "alg"]):
                    d[k] = input[k]
                    if k == 'alg':
                        t[k] = {'alg' : 'CHARSTR', 'doc': 'CHARSTR', version: ['INT32', 3]}
                    else:
                        t[k] = "CHARSTR"
                    continue
                (d2, t2) = self.create_json(input[k])
                d[k] = d2
                t[k] = t2
        elif isinstance(input, list):
            d = []
            t = {}
            for (k, v) in enumerate(input):
                (d2, t2) = self.create_json(v)
                d.append(d2)
                self.merge_dict(t, t2)
        elif isinstance(input, np.ndarray):
            typ = nptypedict[input.dtype]
            if typ[1]:
                d = [float(x) for x in input.ravel()]
            else:
                d = [int(x) for x in input.ravel()]
            t = list((typ[0],)+ input.shape)
        elif isinstance(input, tuple):
            d = input[1]
            t = input[0]
        return (d, t)

    def get(self, name, withtype=False):
        if len(name) == 0:
            return None
        n = splitname(name)
        if n is None:
            return None
        d = self.dict
        while len(n) != 0:
            if isinstance(d, list):
                try:
                    d = d[int(n[0])]
                    n = n[1:]
                except:
                    return None
            elif isinstance(d, dict):
                try:
                    d = d[n[0]]
                    n = n[1:]
                except:
                    return None
        if isinstance(d, tuple) and not withtype:
            return d[1]
        else:
            return d

    def getenumdict(self, name, reverse=False):
        r = self.get(name, True)
        if isinstance(r, tuple):
            if r[0] in typerange.keys():
                return None
            try:
                if reverse:
                    return {v: k for k, v in self.enumdef[r[0]].items()}
                else:
                    return self.enumdef[r[0]]
            except:
                return None
        else:
            return None

    def getenum(self, name):
        r = self.get(name, True)
        if isinstance(r, tuple):
            if r[0] in typerange.keys():
                return r[1]
            try:
                return next(key for key, value in self.enumdef[r[0]].items() if value == r[1])
            except:
                return r[1]
        else:
            return r

    def checknumlist(self, l):
        for v in l:
            if isinstance(v, list):
                if not self.checknumlist(v):
                    return False
            elif not isinstance(v, numbers.Number):
                return False
        return True

    def define_enum(self, name, value):
        # Validate the value dictionary?!?
        self.enumdef[name] = value
        return True

    def set(self, name, value, type="INT32", override=False, append=False):
        if len(name) == 0:
            return False
        n = splitname(name)
        if n is None:
            return False
        d = self.dict
        # Check the type of value!
        if isinstance(value, numbers.Number):
            if not type in typedict.keys() and not type in self.enumdef.keys():
                return False
            value = (type, value)
            issimple = True
        elif isinstance(value, str):
            if type != "CHARSTR":
                return False
            issimple = True
        elif isinstance(value, np.ndarray):
            issimple = True
        elif isinstance(value, cdict):
            issimple = False
        elif isinstance(value, list):
            if self.checknumlist(value):
                if type in self.enumdef.keys():
                    value = np.array(value, dtype='int32')
                else:
                    value = np.array(value, dtype=typedict[type])
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
                if type in self.enumdef.keys():
                    value = (type, value)
            elif isinstance(value, str):
                value = ("CHARSTR", value)
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

    def writeFile(self, file, headers=True):
        return write_typed_json(file, self.dict, self.enumdef, headers)


########################################################################
#
# The rest of this file is helper functions for typed JSON dictionaries.
#
########################################################################

# A little helper function to pull out type information from a typed
# JSON dictionary.  The second argument is either a fully dotted ('b.0.c')
# or python-style ('b0.c') name.
#
# This returns:
#    A simple type string.
#    An enum type dictionary.
#    A list, the first element of which is a simple type string or enum
#    type dictionary, and the remaining items are the dimensions of the array.
#
def getType(typed_json, name):
    if not isinstance(typed_json, dict):
        raise TypeError("getType: First argument should be a typed JSON dictionary!")
    try:
        t = typed_json[':types:']
        try:
            e = t[':enum:']
        except:
            e = {}
    except:
        t = None
    v = typed_json
    n = splitname(name)
    for i in n:
        try:
            v = v[i]
            if t is None:
                if ':types:' not in v.keys():
                    raise TypeError("getType: First argument should be a typed JSON dictionary!")
                t = v[':types:']
                try:
                    e = t[':enum:']
                except:
                    e = {}
            elif not isinstance(i, numbers.Number):
                t = t[i]
        except TypeError:
            raise
        except:
            return None
    if isinstance(t, list):
        if t[0] in typerange.keys():
            return t
        if t[0] in e.keys():
            t = list(t) # Make a copy!
            t[0] = e[t[0]]
            return t
        raise TypeError("getType: Invalid array type %s" % t[0])
    else:
        if t in typerange.keys():
            return t
        if t in e.keys():
            return e[t]
        raise TypeError("getType: Invalid type %s" % t)

#
# Get the value from a typed JSON dictionary.
#
def getValue(typed_json, name):
    if not isinstance(typed_json, dict):
        raise TypeError("getType: First argument should be a typed JSON dictionary!")
    v = typed_json
    n = splitname(name)
    for i in n:
        try:
            v = v[i]
        except:
            return None
    return v

#
# Convert a string, v, to a value of the specified simple type, t.  e is an
# enum dictionary.  Arrays need not apply.
#
def simpleConvert(v, t, e):
    if t in e.keys():
        if v in e[t].keys():
            return e[t][v]
        elif int(v)in e[t].values():
            return int(v)
        else:
            raise TypeError("convertValue: %s is not a valid enum of %s" % (v, t))
    elif t == 'CHARSTR':
        return v
    elif t == 'FLOAT' or t == 'DOUBLE':
        return float(v)
    elif t in typedict.keys():
        v = int(v)
        if v >= typerange[t][0] and v <= typerange[t][1]:
            return v
        else:
            raise ValueError("convertValue: %s is not in range of %s!" % (v, t))
    else:
        raise TypeError("convertValue: %s is not a valid type specifier." % t)

def convertValue(v, t, e={}):
    if isinstance(t, str):
        return simpleConvert(v, t, e)
    elif isinstance(t, list):
        vs = v.split(' ')
        l = np.prod(t[1:])
        if len(vs) != l:
            raise TypeError("convertValue: value has %d elements, not %d!" % (len(vs), l))
        return [simpleConvert(vw, t[0], e) for vw in vs]
    else:
        raise TypeError("convertValue: type must be a str or list.")

#
# Store new values into a typed JSON dictionary.  The value here is always
# a string.  If we have an array value, it will be a space-separated list
# of values.
#
# Returns an integer status:
#     =0 - ok
#     =1 - non-existent path
#     =2 - type conversion failed
#     =3 - invalid dictionary
#
def updateValue(typed_json, name, value):
    if not isinstance(typed_json, dict):
        return 3
    try:
        t = typed_json[':types:']
        try:
            e = t[':enum:']
        except:
            e = {}
    except:
        t = None
    v = typed_json
    n = splitname(name)
    ln = n[-1]
    n = n[:-1]
    for i in n:
        try:
            v = v[i]
            if t is None:
                if ':types:' not in v.keys():
                    return 3
                t = v[':types:']
                try:
                    e = t[':enum:']
                except:
                    e = {}
            elif not isinstance(i, numbers.Number):
                t = t[i]
        except:
            return 1
    # t and v better be dictionaries with an entry ln!
    if (not isinstance(v, dict) or not isinstance(t, dict) or 
        ln not in v.keys() or ln not in t.keys()):
        return 1
    try:
        v[ln] = convertValue(value, t[ln], e)
        return 0
    except:
        return 2
