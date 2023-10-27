import os
from psana.psexp import Run, mode, DataSourceBase
from psana.smalldata import SmallData
from psana.psexp.zmq_utils import zmq_send
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
        # prepare comms for running SmallData
        self.smalldata_obj = SmallData(**self.smalldata_kwargs)
        # send run info to psplotdb server
        if "psmon_publish" in kwargs:
            KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "psplot_live")
            KAFKA_BOOTSTRAP_SERVER = os.environ.get("KAFKA_BOOTSTRAP_SERVER", "172.24.5.240:9094")
            publish = kwargs["psmon_publish"]
            publish.init()
            # Connect to kafka server
            producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVER, 
                    value_serializer=lambda m:json.JSONEncoder().encode(m).encode('utf-8'))
            # Send fully qualified hostname
            fqdn_host = socket.getfqdn()
            info = {'node': fqdn_host,
                    'exp': kwargs['exp'],
                    'runnum': kwargs['run'],
                    'port':publish.port,
                    'slurm_job_id':os.environ.get('SLURM_JOB_ID', os.getpid())}
            producer.send(KAFKA_TOPIC, info)
            #PSPLOT_LIVE_ZMQ_SERVER = os.environ.get("PSPLOT_LIVE_ZMQ_SERVER", "")
            #if PSPLOT_LIVE_ZMQ_SERVER == "":
            #    print(f'Cannot connect to psplot_live through zmq. PSPLOT_LIVE_ZMQ_SERVER not defined')
            #else:
            #    zmq_send(fake_dbase_server=PSPLOT_LIVE_ZMQ_SERVER, 
            #            node=fqdn_host, 
            #            exp=kwargs['exp'], 
            #            runnum=kwargs['run'], 
            #            port=publish.port,
            #            slurm_job_id=os.environ.get('SLURM_JOB_ID', os.getpid()))

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
