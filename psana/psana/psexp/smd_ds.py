class SmdDataSource():

    def __init__(self, configs, eb, run=None):
        self.configs = configs
        self.eb = eb
        self.run = run

    def events(self):
        for i in range(3):
            yield i

        self.eb.build()
        while self.eb.nevents > 0:
            self.eb.build()
            yield Event._from_bytes(self.configs, self.eb.evt_bytes, run=self.run) 


