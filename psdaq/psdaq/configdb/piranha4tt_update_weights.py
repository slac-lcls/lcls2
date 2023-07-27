import psdaq.configdb.tt_update_weights as ttuw
from psdaq.configdb.piranha4tt_config_store import piranha4tt_cdict

if __name__ == "__main__":
    ttuw.main('piranha4', piranha4tt_cdict)
