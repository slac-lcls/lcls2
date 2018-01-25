import zmq
import subprocess



class ProcessManager:
    def __init__(self):
        self.context = zmq.Context()
        self.port = 5556
        self.processes = []
        for i in range(2):
            self.processes.append(self.create_process())

    def create_process(self):
        socket = self.context.socket(zmq.PAIR)
        socket.bind('tcp://*:%d' % (self.port))
        p = subprocess.Popen(['python', 'area_window.py', str(self.port)])
        self.port += 1
        return (p, socket)

    def get_socket(self):
        socket = self.processes.pop()[1]
        self.processes.append(self.create_process())
        return socket
