import zmq
import time

class Barrier:
    """
    From https://zguide.zeromq.org/docs/chapter2/#Node-Coordination
    """
    def __init__(self):
        self.context = zmq.Context()
        self.nworker = 0
        self.publisher = False
        self.subscriber = False
        self.supervisor = False

    def __del__(self):
        self.shutdown()

    def init(self, supervisor, nworker, port1=5561, port2=5562):
        self.supervisor = supervisor
        self.nworker = nworker
        self.port1 = port1
        self.port2 = port2
        if self.nworker==0: return # do nothing if only one opal
        if self.supervisor:
            self.supervisor_init()
        else:
            self.worker_init()

    def shutdown(self):
        # To be called on the deallocate transition
        if self.nworker==0: return
        if self.supervisor:
            self.publisher.close()
            self.syncservice.close()
        else:
            self.subscriber.close()
            self.syncworker.close()
        self.nworker = 0
        self.publisher = False
        self.subscriber = False

    def supervisor_init(self):
        # Socket to talk to workers
        if not self.publisher:
            self.publisher = self.context.socket(zmq.PUB)
            # set SNDHWM, so we don't drop messages for slow subscribers
            # cpo: not necessary for this example since we're sending slow/small msgs
            #self.publisher.sndhwm = 1100000
            self.publisher.bind(f"tcp://*:{self.port1}")
            # Socket to receive signals
            self.syncservice = self.context.socket(zmq.REP)
            self.syncservice.bind(f"tcp://*:{self.port2}")

        # Get synchronization from subscribers
        subscribers = 0
        while subscribers < self.nworker:
            # wait for synchronization request
            msg = self.syncservice.recv()
            # send synchronization reply
            self.syncservice.send(b'')
            subscribers += 1

    def worker_init(self):
        if not self.subscriber:
            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.connect(f"tcp://localhost:{self.port1}")
            self.subscriber.setsockopt(zmq.SUBSCRIBE, b'')

            """
            From https://zguide.zeromq.org/docs/chapter2/#Node-Coordination
            We can’t assume that the SUB connect will be finished
            by the time the REQ/REP dialog is complete. There are
            no guarantees that outbound connects will finish in any
            order whatsoever, if you’re using any transport except
            inproc. So, the example does a brute force sleep of one
            second between subscribing, and sending the REQ/REP synchronization.
            """
            time.sleep(1)

            # Second, synchronize with publisher
            self.syncworker = self.context.socket(zmq.REQ)
            self.syncworker.connect(f"tcp://localhost:{self.port2}")

        # send a synchronization request
        self.syncworker.send(b'')

        # wait for synchronization reply
        self.syncworker.recv()

    def wait(self):
        if self.nworker==0: return # do nothing if only one opal
        if self.supervisor:
            self.publisher.send(b"unblock")
        else:
            self.subscriber.recv()

if __name__ == "__main__":

    import sys
    import time
    supervisor = sys.argv[1]=='s'
    nworker = int(sys.argv[2])
    port1 = 5561
    port2 = 5562

    barrier = Barrier()
    barrier.init(supervisor,nworker,port1,port2)
    for i in range(5):
        if supervisor: time.sleep(1) # supervisor does some work
        barrier.wait() # allow workers to continue
        print('done',supervisor)
