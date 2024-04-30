from main_cnf import *
from psdaq.slurm.config import Config

config = Config(main_config)
config.select(['timing_0', 'control', 'teb0'])

config.add({id:'control_gui', 
    flags:'p', 
    cmd:f'control_gui --uris {cdb} --expert {auth} --loglevel WARNING'})
#config.rename(['timing_0', 'timing_1'], ['teb0', 'teb1'])
#config.show()


