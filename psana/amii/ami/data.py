import time
import numpy as np
from enum import Enum


class MsgTypes(Enum):
    Transition = 0
    Occurrence = 1
    Datagram = 2

class DataTypes(Enum):
    Unknown = 0
    Scalar = 1
    Waveform = 2
    Image = 3

class Transitions(Enum):
    Allocate = 0
    Configure = 1
    Enable = 2
    Disable = 3

class Occurrences(Enum):
    Heartbeat = 0
    User = 1

class Transition(object):
    def __init__(self, ttype, payload):
        self.ttype = ttype
        self.payload = payload

class Datagram(object):
    def __init__(self, name, dtype, data=None):
        self.name = name
        self.dtype = dtype
        self.__data = data

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data

    def __str__(self):
        return "Datagram:\n dtype: %s\n data: %s"%(self.dtype, self.data)

class Message(object):
    def __init__(self, mtype, payload):
        self.mtype = mtype
        self.payload = payload

class StaticSource(object):
    def __init__(self, idnum, interval, init_time, heartbeat, config):
        np.random.seed([idnum])
        self.interval = interval
        self.heartbeat = heartbeat
        self.init_time = init_time
        self.config = config

    def partition(self):
        return [ (key, getattr(DataTypes, value['dtype'])) for key, value in self.config.items() ]

    def events(self):
        count = 0
        emit = False
        time.sleep(self.init_time)
        while True:
            if emit:
                emit = False
                yield Message(MsgTypes.Occurrence, Occurrences.Heartbeat)
            else:
                event = []
                for name, config in self.config.items():
                    if config['dtype'] == 'Scalar':
                        event.append(Datagram(name, getattr(DataTypes, config['dtype']), config['range'][0] + (config['range'][1] - config['range'][0]) * np.random.rand(1)[0]))
                    elif config['dtype'] == 'Waveform' or config['dtype'] == 'Image':
                        event.append(Datagram(name, getattr(DataTypes, config['dtype']), np.random.normal(config['pedestal'], config['width'], config['shape'])))
                    else:
                        print("DataSrc: %s has unknown type %s", name, config['dtype'])
                count += 1
                emit = (count % self.heartbeat == 0)
                yield Message(MsgTypes.Datagram, event)
            time.sleep(self.interval)
        
