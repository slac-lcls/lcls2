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
        if "psmon_publish" in kwargs and "psplotdb_server" in kwargs:
            publish = kwargs["psmon_publish"]
            publish.init()
            # Connect to kafka server
            producer = KafkaProducer(bootstrap_servers=kwargs["psplotdb_server"], 
                    value_serializer=lambda m:json.JSONEncoder().encode(m).encode('utf-8'))
            info = {'node': socket.gethostname(),
                    'exp': kwargs['exp'],
                    'runnum': kwargs['run'],
                    'port':publish.port,
                    'slurm_job_id':os.environ.get('SLURM_JOB_ID', os.getpid())}
            producer.send("monatest", info)
            #zmq_send(fake_dbase_server=kwargs["psplotdb_server"], 
            #        node=socket.gethostname(), 
            #        exp=kwargs['exp'], 
            #        runnum=kwargs['run'], 
            #        port=publish.port,
            #        slurm_job_id=os.environ.get('SLURM_JOB_ID', os.getpid()))

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
