from pymongo import *
from psana.dgramPort.typed_json import *
import psana.dgramPort.configdb as cdb
import subprocess, os, time
import pytest
import pprint

class mongo_configdb(object):
    def __init__(self):
        self.mongod = None
        self.port = None
        self.server = None

    def __enter__(self):
        print("Starting mongod!")
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        self.port = 39999
        done = False
        while not done:
            self.port = self.port + 1
            f = open("/dev/null", "w")
            self.mongod = subprocess.Popen([dir_path + "/mongodrun", str(self.port)],
                                           stdout=f, stderr=f)
            try:
                self.mongod.wait(timeout=1)
            except:
                # It didn't die, we timed out!  Yay team!!
                done = True
        self.server = 'localhost:%d' % self.port
        print("Initializing mongod!")
        c = MongoClient("mongodb://" + self.server)
        config = {'_id': 'test', 'members': [{'_id': 0, 'host': self.server}]}
        c.admin.command("replSetInitiate", config)
        c.close()
        time.sleep(5)  # This takes time.  This seems like enough, 1 is too little.
        return self

    def __exit__(self, type, value, tb):
        print("Killing mongod!")
        self.mongod.kill()
        self.mongod.wait()
        return False

class Test_CONFIGDB:
    def test_one(self):
        return True    # I'll fix this tomorrow!
        with mongo_configdb() as mdb:
            c = cdb.configdb(mdb.server, "AMO", True)
            c.add_alias("BEAM")                 # 0
            c.add_alias("NOBEAM")               # 1
            c.add_device_config("evr")
            c.add_device_config("evrIO")
            print("Configs:")
            c.print_configs()
            print("\nDevice configs:")
            c.print_device_configs()
            print("\nevr:")
            c.print_device_configs("evr")
            print("\nevrIO:")
            c.print_device_configs("evrIO")
            d = {}
            e = cdict()
            e.set("a", 32)
            e.set("b", 74)
            d["evr"] = e
            e = cdict()
            e.set("c", 93)
            e.set("d", 67)
            e.set("e", [3, 6.5, 9.4], "DOUBLE")
            d["evrIO"] = e
            c.modify_device("BEAM", "evr0", d)   # 2
            with pytest.raises(Exception):
                c.modify_device("BEAM", "evr0", d)
            d["evrIO"].set("c", 103)
            c.modify_device("NOBEAM", "evr0", d) # 3
            ans_key     = {'BEAM': 2, 'NOBEAM': 3}
            ans_evrIO_c = {'BEAM': 93, 'NOBEAM': 103}
            for a in c.get_aliases():
                k = c.get_key(a)
                assert k == ans_key[a]
                print("\n%s: get_config(%d, evr0):" % (a, k))
                cfg = c.get_configuration(k, "evr0")
                pprint.pprint(cfg)
                assert cfg['evrIO']['c'] == ans_evrIO_c[a]
            d["evrIO"].set("d", 73)
            c.modify_device("BEAM", "evr0", d)   # 4
            pprint.pprint(c.get_configuration("BEAM", "evr0"))
            h = c.get_history("BEAM", "evr0", ["evr.a", "evrIO.d", "evrIO.e"])
            pprint.pprint(h)
            assert len(h) == 2
            assert [67, 73] == [d['evrIO.d'] for d in h]
            assert [2, 4] == [d['key'] for d in h]
            c2 = cdb.configdb(mdb.server, "SXR", True)
            c2.add_alias("FOO")                                       # 0
            c2.transfer_config("AMO", "BEAM", "evr0", "FOO", "evr3")  # 1
            with pytest.raises(Exception):
                c2.transfer_config("AMO", "BEAM", "evr0", "FOO", "evr3")
            print("Configs:")
            c2.print_configs()
            cfg =  c.get_configuration("BEAM", "evr0")
            cfg2 = c2.get_configuration("FOO", "evr3")
            assert cfg == cfg2

def run():
    test = Test_CONFIGDB()
    test.test_one()

if __name__ == "__main__":
    run()
