import psdaq.configdb.tt_update_weights as ttuw
from psdaq.configdb.opaltt_config_store import opaltt_cdict

if __name__ == "__main__":
    ttuw.main('opal', opaltt_cdict)
