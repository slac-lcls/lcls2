import pyrogue as pr

class TimingRx(pr.Device):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add(pr.RemoteVariable(
            name        = 'RxReset',
            description = 'Reset timing receive link',
            offset      = 0x20,
            bitSize     = 1,
            bitOffset   = 3,
            mode        = 'WO',
            verify      = False,
            ))
