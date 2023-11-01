import os
from psana.psexp import Run, mode, DataSourceBase
from psana.smalldata import SmallData
from psana.psexp.zmq_utils import ClientSocket
from kafka import KafkaProducer
import json
import socket

class NullRun(object):
    def __init__(self):
        self.expt = None
        self.runnum = None
        self.epicsinfo = {}
        self.detinfo = {}
    def Detector(self, *args):
        return None
    def events(self):
        return iter([])
    def steps(self):
        return iter([])

class NullDataSource(DataSourceBase):

    def __init__(self, *args, **kwargs):
        super(NullDataSource, self).__init__(**kwargs)
        # Prepare comms for running SmallData
        self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        # Send run info to psplotdb server using kafka (default).
        # Note that you can use zmq instead by specifying zmq server
        # in the env var. below.
        PSPLOT_LIVE_ZMQ_SERVER = os.environ.get("PSPLOT_LIVE_ZMQ_SERVER", "")
        if "psmon_publish" in kwargs:
            publish = kwargs["psmon_publish"]
            publish.init()
            # Send fully qualified hostname
            fqdn_host = socket.getfqdn()
            info = {'node': fqdn_host,
                    'exp': kwargs['exp'],
                    'runnum': kwargs['run'],
                    'port':publish.port,
                    'slurm_job_id':os.environ.get('SLURM_JOB_ID', os.getpid())}
            if PSPLOT_LIVE_ZMQ_SERVER == "":
                KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "psplot_live")
                KAFKA_BOOTSTRAP_SERVER = os.environ.get("KAFKA_BOOTSTRAP_SERVER", "172.24.5.240:9094")
                # Connect to kafka server
                producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVER, 
                        value_serializer=lambda m:json.JSONEncoder().encode(m).encode('utf-8'))
                producer.send(KAFKA_TOPIC, info)
            else:
                sub = ClientSocket(PSPLOT_LIVE_ZMQ_SERVER)
                info['msgtype'] = MonitorMsgType.PSPLOT
                sub.send(info)

    def runs(self):
        yield NullRun()
    
    def is_mpi(self):
        return False

    def unique_user_rank(self):
        """ NullDataSource is used for srv nodes, therefore not a
        'user'-unique rank."""
        return False

    def is_srv(self):
        return True
