import six
import zmq
import threading
from ami import operation


class Worker(object):
    def __init__(self, cfg_host, cfg_port, node_slice=0, num_slices=1, gather=False):
        self.cfg_host = cfg_host
        self.cfg_port = cfg_port
        self.node_slice = node_slice
        self.num_slices = num_slices
        self.gather = gather
        self.config = {}
        self.ops = {}
        self.context = zmq.Context()
        self.cfg_sock = self.context.socket(zmq.SUB)
        self.cfg_sock.setsockopt_string(zmq.SUBSCRIBE, u"config\0")
        self.cfg_sock.connect("tcp://%s:%d"%(cfg_host, cfg_port))
        self.cfg_lock = threading.Lock()

    def check_src_type(self, op):
        if self.gather:
            return isinstance(op, operation.GRBase)
        else:
            return isinstance(op, operation.ExternDataSrc)

    def do_config(self, op):
        if self.gather:
            return not isinstance(op, operation.ExternDataSrc)
        else:
            return not isinstance(op, operation.GRBase)

    def run(self):
        while True:
            changed_ops = []
            srcs = []
            print("Worker running, waiting for cfg...")
            cfg_topic = self.cfg_sock.recv_string()
            config_id = self.cfg_sock.recv_pyobj()
            new_global_config = self.cfg_sock.recv_pyobj()
            print("Recieved new configuration id: %d"%config_id)
            self.cfg_lock.acquire()
            print("Acquired lock - starting reconfig")
            for opid, data in six.iteritems(new_global_config):
                if opid not in self.config:
                    opclass = getattr(operation,  data['optype'])
                    self.ops[opid] = opclass(opid, self.ops)
                    if hasattr(self.ops[opid], 'node_slice'):
                        self.ops[opid].node_slice = self.node_slice
                    if hasattr(self.ops[opid], 'num_slices'):
                        self.ops[opid].num_slices = self.num_slices
                    changed_ops.append(self.ops[opid])
                elif (data != self.config[opid]) or (self.gather and isinstance(self.ops[opid], operation.GRBase)):
                    changed_ops.append(self.ops[opid])
                if self.check_src_type(self.ops[opid]):
                    srcs.append(self.ops[opid])
            for op in changed_ops:
                if self.do_config(op):
                    op._configure(config_id, new_global_config)
            # update all remaining conf ids
            for op in six.itervalues(self.ops):
                op.config_id = config_id
            self.config = new_global_config
            self.cfg_lock.release()
            # Should we support more than one externally data source per worker?
            if len(srcs) > 1:
                print("Currently only on external data source per worker is supported")
                break
            for src in srcs:
                src.start(self.cfg_lock)
            print("Finished configuration of id %d"%config_id)
