from psdaq.configdb.typed_json import *
from psana import DataSource, container
import numpy as np
import subprocess
import os, re

class Test_JSON2XTC:
    def check(self, co, cl, base=""):
        result = True
        for n in dir(co):
            if n[0] != '_':
                v = co.__getattribute__(n)
                # Change xxx_nnn to xxx.nnn
                m = re.search('^(.*)_([0-9]+)$', n)
                if m is not None:
                    n = m.group(1) + "." + m.group(2)
                if base == "":
                    bnew = n
                else:
                    bnew = base + "." + n
                print(bnew, v)
                v2 = cl.get(bnew)
                if isinstance(v, container.Container):
                    if isinstance(v2, int):
                        # bnew must be the name of an enum!
                        if v.value != v2 or v.names != cl.getenumdict(bnew, reverse=True):
                            print("Failure on %s" % bnew)
                            print(v.value)
                            print(v.names)
                            print(v2)
                            result = False
                    else:
                        result = self.check(v, cl, bnew) and result
                elif isinstance(v, np.ndarray):
                    if not (v == v2).all():
                        print("Failure on %s" % bnew)
                        print(v)
                        print(v2)
                        result = False
                elif isinstance(v, int) or isinstance(v, str):
                    if v != v2:
                        print("Failure on %s" % bnew)
                        print(v)
                        print(v2)
                        result = False
                else:
                    if abs(v - v2) > 0.00001:
                        print("Failure on %s" % bnew)
                        print(v)
                        print(v2)
                        result = False
        return result

    def test_one(self, tmp_path):
        c = cdict()
        c.setInfo("test", "test1", None, "serial1234", "No comment")
        c.setAlg("raw", [1,2,3])
        # Scalars
        c.set("aa", -5, "INT8")
        c.set("ab", -5192, "INT16")
        c.set("ac", -393995, "INT32")
        c.set("ad", -51000303030, "INT64")
        c.set("ae", 3, "UINT8")
        c.set("af", 39485, "UINT16")
        c.set("ag", 1023935, "UINT32")
        c.set("ah", 839394939393, "UINT64")
        c.set("ai", 52.4, "FLOAT")
        c.set("aj", -39.455, "DOUBLE")
        c.set("ak", "A random string!", "CHARSTR")
        # Arrays
        c.set("al", [1, 3, 7])
        c.set("am", [[4,6,8],[7,8,3]], "INT16")
        c.set("an", np.array([12, 13, 17],dtype='int32'))
        c.set("ao", np.array([[6,8,13],[19,238,998]],dtype='uint16'))
        # Submodules
        d = cdict()
        d.set("b", 33)
        d.set("c", 39)
        d.set("d", 696)
        c.set("b", d, append=True)
        d.set("b", 95)
        d.set("c", 973)
        d.set("d", 939)
        c.set("b", d, append=True)
        d.set("d", 15)
        c.set("b", d, append=True)
        # Deeper names
        c.set("c.d.e", 42.49, "FLOAT")
        c.set("c.d.f", -30.34, "DOUBLE")
        c.set("b.0.c", 72)
        # Enums
        c.define_enum("q", {"Off": 0, "On": 1})
        c.define_enum("r", {"Down": -1, "Up": 1})
        c.set("c.d.g", 0, "q")
        c.set("c.d.h", 1, "q")
        c.set("c.d.k", 1, "r")
        c.set("c.d.l", -1, "r")
        # MCB - Right now, psana doesn't support arrays of enums, so comment
        # this out.  If CPO ever implements this, this test will need to be
        # revisited.
        # assert c.set("c.d.m", [0, 1, 0], "q")
        # Write it out!!
        json_file = os.path.join(tmp_path, "json2xtc_test.json")
        xtc_file = os.path.join(tmp_path, "json2xtc_test.xtc2")
        assert c.writeFile(json_file)
        # Convert it to xtc2!

        # this test should really run in psdaq where json2xtc
        # lives. as a kludge, abort the test if the exe is
        # not found - cpo
        exe_found = False
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, 'json2xtc')
            if os.path.isfile(exe_file):
                exe_found = True
                break
        if not exe_found: return

        subprocess.call(["json2xtc", json_file, xtc_file])
        # Now, let's read it in!
        ds = DataSource(files=xtc_file)
        dg = ds._configs[0]
        print(dir(dg.software), c.dict['detId:RO'])
        assert dg.software.test1[0].detid == c.dict['detId:RO']
        assert dg.software.test1[0].dettype == c.dict['detType:RO']
        assert self.check(dg.test1[0].raw, c)

def run():
    test = Test_JSON2XTC()
    import pathlib
    test.test_one(pathlib.Path('.'))

if __name__ == "__main__":
    run()
