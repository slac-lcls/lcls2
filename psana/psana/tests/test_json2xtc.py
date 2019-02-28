from psana.dgramPort.typed_json import *
from psana import DataSource
import numpy as np
import subprocess
import os

class Test_JSON2XTC:
    @classmethod
    def setup_class(cls):
        for f in ["json2xtc_test.json", "json2xtc_test.xtc2"]:
            try:
                os.remove(f)
            except:
                pass

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
        assert c.set("c_d_e", 42.49, "FLOAT")
        assert c.set("c_d_f", -30.34, "DOUBLE")
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
        segment = 0
        for n in dir(dg.test1[segment].raw):
            if n[0] != '_':
                if isinstance(c.get(n), np.ndarray):
                    if not (dg.test1[segment].raw.__getattribute__(n) == c.get(n)).all():
                        print("Failure on %s" % n)
                        print(dg.test1[segment].raw.__getattribute__(n))
                        print(c.get(n))
                        assert False
                elif isinstance(c.get(n), int) or isinstance(c.get(n), str):
                    if not dg.test1[segment].raw.__getattribute__(n) == c.get(n):
                        print("Failure on %s" % n)
                        print(dg[0].test1[segment].raw.__getattribute__(n))
                        print(c.get(n))
                        assert False
                else:
                    if not abs(dg.test1[segment].raw.__getattribute__(n) - c.get(n)) < 0.00001:
                        print("Failure on %s" % n)
                        print(dg.test1[segment].raw.__getattribute__(n))
                        print(c.get(n))
                        assert False

def run():
    test = Test_JSON2XTC()
    test.test_one()

if __name__ == "__main__":
    run()
