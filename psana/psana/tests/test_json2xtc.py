from psalg.configdb.typed_json import *
from psana import DataSource, container
import numpy as np
import subprocess
import os, re

class Test_JSON2XTC:
    @classmethod
    def setup_class(cls):
        for f in ["json2xtc_test.json", "json2xtc_test.xtc2"]:
            try:
                os.remove(f)
            except:
                pass

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

    def test_one(self):
        c = cdict()
        c.setInfo("test", "test1", "serial1234", "No comment")
        c.setAlg("raw", [1,2,3])
        # Scalars
        assert c.set("aa", -5, "INT8")
        assert c.set("ab", -5192, "INT16")
        assert c.set("ac", -393995, "INT32")
        assert c.set("ad", -51000303030, "INT64")
        assert c.set("ae", 3, "UINT8")
        assert c.set("af", 39485, "UINT16")
        assert c.set("ag", 1023935, "UINT32")
        assert c.set("ah", 839394939393, "UINT64")
        assert c.set("ai", 52.4, "FLOAT")
        assert c.set("aj", -39.455, "DOUBLE")
        assert c.set("ak", "A random string!", "CHARSTR")
        # Arrays
        assert c.set("al", [1, 3, 7])
        assert c.set("am", [[4,6,8],[7,8,3]], "INT16")
        assert c.set("an", np.array([12, 13, 17],dtype='int32'))
        assert c.set("ao", np.array([[6,8,13],[19,238,998]],dtype='uint16'))
        # Submodules
        d = cdict()
        assert d.set("b", 33)
        assert d.set("c", 39)
        assert d.set("d", 696)
        assert c.set("b", d, append=True)
        assert d.set("b", 95)
        assert d.set("c", 973)
        assert d.set("d", 939)
        assert c.set("b", d, append=True)
        assert d.set("d", 15)
        assert c.set("b", d, append=True)
        # Deeper names
        assert c.set("c.d.e", 42.49, "FLOAT")
        assert c.set("c.d.f", -30.34, "DOUBLE")
        assert c.set("b.0.c", 72)
        # Enums
        assert c.define_enum("q", {"Off": 0, "On": 1})
        assert c.define_enum("r", {"Down": -1, "Up": 1})
        assert c.set("c.d.g", 0, "q")
        assert c.set("c.d.h", 1, "q")
        assert c.set("c.d.k", 1, "r")
        assert c.set("c.d.l", -1, "r")
        # MCB - Right now, psana doesn't support arrays of enums, so comment
        # this out.  If CPO ever implements this, this test will need to be
        # revisited.
        # assert c.set("c.d.m", [0, 1, 0], "q")
        # Write it out!!
        assert c.writeFile("json2xtc_test.json")
        # Convert it to xtc2!
        subprocess.call(["json2xtc", "json2xtc_test.json", "json2xtc_test.xtc2"])
        # Now, let's read it in!
        ds = DataSource("json2xtc_test.xtc2")
        myrun = next(ds.runs())
        dg = myrun.configs[0]
        assert dg.software.test1.detid == c.dict['detId']
        assert dg.software.test1.dettype == c.dict['detType']
        assert self.check(dg.test1[0].raw, c)

def run():
    test = Test_JSON2XTC()
    test.test_one()

if __name__ == "__main__":
    run()
