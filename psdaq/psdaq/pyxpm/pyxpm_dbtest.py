from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
from psdaq.configdb.get_config import get_config_with_params
import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for XPM')

parser.add_argument('-P', required=True, help='e.g. DAQ:LAB2:XPM:1', metavar='PREFIX')
parser.add_argument('--db', type=str, default=None, help="save/restore db, for example [https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/,configDB,LAB2,PROD]")

args = parser.parse_args()

init = None

name = args.P
db = args.db
db_url, db_name, db_instrument, db_alias = db.split(',',4)
print('db {:}'.format(db))
print('url {:}  name {:}  instr {:}  alias {:}'.format(db_url,db_name,db_instrument,db_alias))
print('device {:}'.format(name))
dbinit = get_config_with_params(db_url, db_instrument, db_name, db_alias, name)
print('cfg {:}'.format(dbinit))

print(dbinit['XTPG'])
print(dbinit['XTPG']['CuDelay'])
cuDelay    = dbinit['XTPG']['CuDelay']
print(dbinit['XTPG']['CuBeamCode'])
cuBeamCode = dbinit['XTPG']['CuBeamCode']
print(dbinit['XTPG']['CuInput'])
cuInput    = dbinit['XTPG']['CuInput']
print('Read XTPG parameters CuDelay {}, CuBeamCode {}, CuInput {}'.format(cuDelay,cuBeamCode,cuInput))

