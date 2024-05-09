import copy, os


class Config:
    """Allows users to modify main-config list"""

    def __init__(self, main_config):
        # Turns main config list into a dictionary for ez lookup
        self.main_config = {}
        for _config in main_config:
            config = copy.deepcopy(_config)
            cid = config["id"]
            config.pop("id")
            self.main_config[cid] = config
        self.select_config = {}

    def select(self, items):
        """Select only configs with given config ids or add
        new one if given a list of dictionary.

        items example:
        ids: ['timing_0', 'teb0', 'control'] or
        configs: [{ host: ...}, { host: ...}, ...]
        """
        for item in items:
            if isinstance(item, dict):
                config = copy.deepcopy(item)
                cid = config["id"]
                config.pop("id")
                self.select_config[cid] = config
            else:
                cid = item
                if cid in self.main_config:
                    self.select_config[cid] = copy.deepcopy(self.main_config[cid])
                else:
                    print(f"Warning: no {cid} found in main_config")

    def add(self, config):
        cid = config["id"]
        config.pop("id")
        self.select_config[cid] = config

    def rename(self, *args):
        for arg in args:
            current_cid, new_cid = arg
            if current_cid not in self.select_config:
                self.select([current_cid])
            if current_cid in self.select_config:
                self.select_config[new_cid] = self.select_config.pop(current_cid)
            else:
                print(f"Warning: no {current_cid} found in select_config")

    def show(self, full=False):
        print("%20s %18s %80s" % ("UniqueID", "Host", "Command+Args"))
        for cid, config_detail in self.select_config.items():
            host = "localhost"
            if "host" in config_detail:
                host = config_detail["host"]
            cmd = config_detail["cmd"][:65] + "..." + config_detail["cmd"][-12:]
            if full:
                cmd = config_detail["cmd"]
            print("%20s %18s %80s" % (cid, host, cmd))
