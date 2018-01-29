import zmq
import time
from ami.operation.base import Operation


class Print(Operation):
    def __init__(self, opid, ops):
        super(Print, self).__init__(opid, ops)
        self.add_input('info')

    def run(self):
        print(self.event_id, self.config_id, self.info)
        return True


class Rate(Operation):
    def __init__(self, opid, ops):
        super(Rate, self).__init__(opid, ops)
        self.config.require('rate', 'name')
        self.add_input('null')

    def configure(self):
        self.count = 0
        self.last = time.time()
        return True

    def run(self):
        now = time.time()
        self.count += 1
        if now - self.last > self.rate:
            print("Events per second (%s): %.2f"%(self.name, self.count/self.rate))
            self.count = 0
            self.last = now
        return True


class Send(Operation):
    def __init__(self, opid, ops):
        super(Send, self).__init__(opid, ops)
        self.config.require('port', 'topic')
        self.add_input('data')
        self.context = zmq.Context()
        self.send = None

    def configure(self):
        if self.send is None:
            self.send = self.context.socket(zmq.PUB)
            self.send.bind("tcp://*:%d"%self.config['port'])
        return True

    def run(self):
        self.send.send_string(self.config['topic'], zmq.SNDMORE)
        self.send.send_pyobj(self.data)
        return True
    

class Image(Operation):
    def __init__(self, opid, ops):
        super(Image, self).__init__(opid, ops)
        self.config.require('title', 'rate')
        self.add_input('image')

    def configure(self):
        self.last = time.time()
        return True

    def run(self):
        now = time.time()
        if now - self.last > (1/self.rate):
            self.img = plots.Image(time.ctime(), self.title, self.image)
            publish.send(self.opid, self.img)
            self.last = now
        return True
