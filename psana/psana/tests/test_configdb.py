from pymongo import *
from psalg.configdb.typed_json import *
import psalg.configdb.configdb as cdb
import subprocess, os, time, sys
import tempfile
import shutil
import pytest
import pprint

class mongo_configdb(object):
    def __init__(self, server=None, hutch=None, create=False, root="configDB"):
        self.mongod = None
        self.port = None
        self.server = server
        self.hutch = hutch
        self.create = create
        self.standalone = False
        self.root = root
        self.cdb = None
        self.dbpath = None

    def __enter__(self):
        if self.server is None:
            self.port = 39999
            done = False
            # create a temporary directory for dbpath
            self.dbpath = tempfile.mkdtemp()
            while not done:
                self.port = self.port + 1
                f = open("/dev/null", "w")
                self.mongod = subprocess.Popen(['mongod', '--bind_ip_all', '--dbpath', self.dbpath,
                                                '--port', str(self.port), '--logpath', '/dev/null'],
                                                stdout=f, stderr=f)
                # Now, wait for mongod to die or to finish initializing!
                while True:
                    time.sleep(1)
                    if self.mongod.poll() is not None:  # It's dead, Jim.
                        break
                    try:
                        state = subprocess.check_output(["ps", "ho", "state",
                                                         str(self.mongod.pid)]).strip()
                        if sys.version_info.major == 3:
                            state = state.decode()
                    except:
                        state = "D"  # It's probably dead, so just report a not-done status.
                    if state[0] == 'S':
                        done = True
                        break
            self.server = 'localhost:%d' % self.port
            self.standalone = True
            """
            # We don't run with replica sets and transactions any more!
            try:
                c = MongoClient("mongodb://" + self.server)
                config = {'_id': 'test', 'members': [{'_id': 0, 'host': self.server}]}
                c.admin.command("replSetInitiate", config)
                c.close()
                time.sleep(5)  # This takes time.  This seems like enough, 1 is too little.
            except:
                self.mongod.kill()
                self.mongod.wait()
                self.mongod = None
                raise
            """
        else:
            self.standalone = False
        if self.hutch is not None:
            self.cdb = cdb.configdb(self.server, self.hutch, self.create, self.root)
        return self

    def __exit__(self, type, value, tb):
        if self.mongod is not None:
            self.mongod.kill()
            self.mongod.wait()
        if self.cdb is not None and not self.standalone:
            self.cdb.client.drop_database(self.root)
        if self.dbpath is not None:
            shutil.rmtree(self.dbpath)
        return False

class Test_CONFIGDB:
    # server is the mongodb server string.  If it is None (which is the default and will
    # be the value when invoked with pytest), then start a standalone mongod server for the
    # test.
    def test_one(self, server=None):
        print(server)
        dbname = "regress"+str(os.getpid())
        with mongo_configdb(server, "AMO", create=True, root=dbname) as mdb:
            c = mdb.cdb
            c.add_alias("BEAM")                 # 0
            c.add_alias("NOBEAM")               # 1
            c.add_device_config("evr")
            c.add_device_config("jungfrau")
            print("Configs:")
            c.print_configs()
            print("\nDevice configs:")
            c.print_device_configs()
            print("\nevr:")
            c.print_device_configs("evr")
            print("\njungfrau:")
            c.print_device_configs("jungfrau")
            e = cdict()
            e.set("a:RO", 32) # test the "readonly" flag
            e.set("b", 74)
            e.setInfo(detType="evr", detName="evr0")
            j = cdict()
            j.set("c", 93)
            j.set("d", 67)
            j.set("e", [3, 6.5, 9.4], "DOUBLE")
            j.setInfo(detType="jungfrau", detName="jungfrau0")
            c.modify_device("BEAM", e)   # 2
            c.modify_device("BEAM", j)   # 3
            with pytest.raises(Exception):
                c.modify_device("BEAM", e)
            j.set("c", 103)
            c.modify_device("NOBEAM", j) # 4
            ans_key        = {'BEAM': 3,  'NOBEAM': 4}
            ans_jungfrau_c = {'BEAM': 93, 'NOBEAM': 103}
            print("Configs:")
            c.print_configs()
            for a in c.get_aliases():
                k = c.get_key(a)
                assert k == ans_key[a]
                print("\n%s: get_config(%d, jungfrau0):" % (a, k))
                cfg = c.get_configuration(k, "jungfrau0")
                pprint.pprint(cfg)
                assert cfg['c'] == ans_jungfrau_c[a]
            j.set("d", 73)
            c.modify_device("BEAM", j)   # 5
            pprint.pprint(c.get_configuration("BEAM", "jungfrau0"))
            h = c.get_history("BEAM", "jungfrau0", ["d", "e"])
            pprint.pprint(h)
            assert len(h) == 2
            assert [67, 73] == [d['d'] for d in h]
            assert [3, 5] == [d['key'] for d in h]
            c2 = cdb.configdb(mdb.server, "SXR", create=True, root=dbname)
            c2.add_alias("FOO")                                       # 0
            c2.transfer_config("AMO", "BEAM", "evr0", "FOO", "evr3")  # 1
            with pytest.raises(Exception):
                c2.transfer_config("AMO", "BEAM", "evr0", "FOO", "evr3")
            print("Configs:")
            c2.print_configs()
            cfg =  c.get_configuration("BEAM", "evr0")
            cfg2 = c2.get_configuration("FOO", "evr3")
            assert cfg == cfg2
            print("Test complete!")

def run():
    test = Test_CONFIGDB()
    test.test_one("mcbrowne:psana@psdb-dev:9306")
    test.test_one()

if __name__ == "__main__":
    run()
