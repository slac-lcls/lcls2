import time

def pvUpdate(pv, val):
    value = pv.current()
    value['value'] = val
    value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
    pv.post(value)

class DefaultPVHandler(object):

    def __init__(self):
        pass

    def put(self, pv, op):
        postedval = op.value()
        print('DefaultPVHandler.put ',pv,postedval['value'])
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
        pv.post(postedval)
        op.done()

class PVHandler(object):

    def __init__(self,cb):
        self._cb = cb

    def put(self, pv, op):
        postedval = op.value()
        print('PVHandler.put ',postedval['value'],self._cb)
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
        pv.post(postedval)
        self._cb(pv,postedval['value'])
        op.done()

