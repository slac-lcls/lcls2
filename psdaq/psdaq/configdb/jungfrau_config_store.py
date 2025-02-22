from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import sys
import argparse
import IPython
import pyrogue as pr

def lookupValue(d,name):
    key = name.split(".",1)
    if key[0] in d:
        v = d[key[0]]
        if isinstance(v,dict):
            return lookupValue(v,key[1])
        elif isinstance(v,bool):
            return 1 if v else 0
        else:
            return v
    else:
        return None

class mcdict(cdict):
    def __init__(self, fn=None):
        super().__init__(self)

        self._yamld = {}
        if fn:
            print("Loading yaml...")
            self._yamld = pr.yamlToData(fName=fn)

    #  intercept the set call to replace value with yaml definition
    def init(self, prefix, name, value, type="INT32", override=False, append=False):
        v = lookupValue(self._yamld,name)
        if v:
            print("Replace {:}[{:}] with [{:}]".format(name,value,v))
            value = v
        self.set(prefix+":RO."+name+":RO", value, type, override, append)

def write_to_daq_config_db(args):
    create: bool = True
    dbname: str = "configDB"

    db: str  = "configdb" if args.prod else "devconfigdb"
    url: str  = f"https://pswww.slac.stanford.edu/ws-auth/{db}/ws/"

    mycdb = cdb.configdb(url, args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config("jungfrau")

    top: mcdict = mcdict(args.yaml)
    top.setInfo("jungfrau", args.name, args.segm, args.id, "No comment")
    top.setAlg("config", [0,1,0])

    help_str: str = (
        "-- user --\n"
        "  - bias_voltage_v : Module bias voltage in V.\n"
        "  - trigger_delay_s : Additional trigger delay in s.\n"
        "  - exposure_time_s : Exposure time in s.\n"
        "  - exposure_period : Exposure period in s.\n"
        "  - port : Port.\n"
    )
    top.set("help:RO", help_str, "CHARSTR")

    # For Jungfrau module
    top.set("user.bias_voltage_v", 200, "UINT8") # Int??
    top.set("user.trigger_delay_s", 0.000238, "DOUBLE")
    top.set("user.exposure_time_s", 0.00001, "DOUBLE")
    top.set("user.exposure_period", 0.2, "DOUBLE")
    top.set("user.port", 8192, "UINT16")

    top.define_enum(
        "gainModeEnum",
        {
            "DYNAMIC":0,
            "FORCE_SWITCH_G1":1,
            "FORCE_SWITCH_G2":2,
            "FIX_G1":3,
            "FIX_G2":4,
            "FIX_G0":5,
        }
    )
    top.set("user.gainMode", 3, "gainModeEnum")

    top.define_enum("gain0Enum", {"normal":0, "high":1})
    top.set("user.gain0", 0, "gain0Enum")

    top.define_enum(
        "speedLevelEnum",
        {
            "FULL_SPEED": 0,
            "HALF_SPEED": 1,
            "QUARTER_SPEED": 2,
        }
    )
    top.set("user.speedLevel", 1, "speedLevelEnum")

    top.set("user.jungfrau_mac", "aa:bb:cc:dd:ee:ff", "CHARSTR")
    top.set("user.jungfrau_ip", "10.0.0.15", "CHARSTR")

    # For Kcu1500/C1100
    top.set("user.kcu_mac", "08:00:56:00:00:00", "CHARSTR")
    top.set("user.kcu_ip", "10.0.0.10", "CHARSTR")

    top.set("expert.PauseThreshold", 16, "UINT8")
    top.set("expert.TriggerDelay", 42, "UINT32") # 185.7 MHz clocks

    mycdb.add_alias(args.alias)
    mycdb.modify_device(args.alias, top)


if __name__ == "__main__":
    args = cdb.createArgs().args
    write_to_daq_config_db(args)
