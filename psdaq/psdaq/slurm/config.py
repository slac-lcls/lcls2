import copy, os

class Config():
    """Allows users to modify main-config list """
    def __init__(self, main_config):
        # Turns main config list into a dictionary for ez lookup
        self.main_config = {}
        for _config in main_config:
            config = copy.deepcopy(_config)
            cid = config['id']
            config.pop('id')
            self.main_config[cid] = config
        self.select_config = {}

    def select(self, ids):
        """Select only configs with given config ids"""
        for cid in ids:
            if cid in self.main_config:
                self.select_config[cid] = copy.deepcopy(self.main_config[cid])
            else:
                print(f'Warning: no {cid} found in main_config')

    def add(self, config):
        cid = config['id']
        config.pop('id')
        self.select_config[cid] = config 

    def rename(self, *args):
        for arg in args:
            current_cid, new_cid = arg
            if current_cid in self.select_config:
                self.select_config[new_cid] = self.select_config.pop(current_cid)
            else:
                print(f'Warning: no {current_cid} found in select_config')

    def show(self):
        for cid, config_detail in self.select_config.items():
            print(f'{cid} {config_detail}')
            print('')
