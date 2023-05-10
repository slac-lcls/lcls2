import sys
import numpy
import sysv_ipc
import logging
import json
import argparse

from struct import unpack
import EbDgram     as edg
import ResultDgram as rdg


class ArgsParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__()
        self.add_argument('-p', type=int, choices=range(0, 8), default=0, help='partition (default 0)')
        self.add_argument('-b', type=int, required=True, help='IPC key base value')

    def parse(self):
        self.args = self.parse_args()
        return self.args

class TriggerDataSource(object):
    def __init__(self):

        logging.info(f"[Python] Starting")

        self._mq_inp = None
        self._mq_res = None
        self._shm_inp = None
        self._shm_res = None

        self.connect_json = None

        # Make args available to the user scripts
        self.args = ArgsParser().parse()

        # Make the base key value depend on the partition number
        KEY_BASE = self.args.b

        try:
            self._mq_inp = sysv_ipc.MessageQueue(KEY_BASE + 0)
        except sysv_ipc.Error as exp:
            print(
                f"[Python] Error connecting to 'Inputs' message queue - Error: {exp}"
            )
            sys.exit(1)

        try:
            self._mq_res = sysv_ipc.MessageQueue(KEY_BASE + 1)
        except sysv_ipc.Error as exp:
            print(
                f"[Python] Error connecting to 'Results' message queue - Error: {exp}"
            )
            sys.exit(1)

        print(f"[Python] Connected to message queues")

        while True:             # Synch up when there's cruft in the pipe
            message, priority = self._mq_inp.receive()
            print(f"[Python] Received message '{message}', prio '{priority}'")

            if chr(message[0]) != 'i':
                print(f"[Python] Ignoring unrecognized message '{chr(message[0])}'; expected 'i'")
            else:
                try:
                    shm_msg = message.decode().split(',')
                    print(f'P** inp shm_msg: {shm_msg[1]}, {shm_msg[2]}')
                    self._shm_inp = sysv_ipc.SharedMemory(int(shm_msg[1]), size=int(shm_msg[2]))
                    inputsSize = 0
                    self._shm_inp_bufSizes = []
                    for ctrbSize in shm_msg[3:]:
                        self._shm_inp_bufSizes.append(inputsSize)
                        inputsSize += int(ctrbSize)
                    self._shm_inp_bufSizes.append(inputsSize)
                except sysv_ipc.Error as exp:
                    print(
                        f"[Python] Error connecting to 'Inputs' shared memory - Error: {exp}"
                    )
                    sys.exit(1)

                print(f"[Python] Set up Inputs shared memory key {shm_msg[1]}")
                break

        while True:             # Synch up when there's cruft in the pipe
            message, priority = self._mq_inp.receive()
            print(f"[Python] Received message '{message}', prio '{priority}'")

            if chr(message[0]) != 'r':
                print(f"[Python] Unrecognized message '{chr(message[0])}'; expected 'r'")
            else:
                try:
                    shm_msg = message.decode().split(',')
                    self._shm_res = sysv_ipc.SharedMemory(int(shm_msg[1]), size=int(shm_msg[2]))
                    print(f'P** res shm_msg: {shm_msg[1]}, {shm_msg[2]}')
                except sysv_ipc.Error as exp:
                    print(
                        f"[Python] Error connecting to 'Results' shared memory - Error: {exp}"
                    )
                    self._shm_inp.detach()
                    self._shm_inp = None
                    sys.exit(1)

                print(f"[Python] Set up Results shared memory key {shm_msg[1]}")
                break

        #print(f'max_size: {self._mq_inp.max_size}')

        connectMsg = ''
        while True:
            message, priority = self._mq_inp.receive()
            print(f"[Python] Received message '{message[:10]}', len '{len(message)}, prio '{priority}'")

            if   chr(message[0]) in ('c', 'd'):
                size        = int((message.decode())[2:])
                connectMsg += self._shm_inp.read(size).decode()
                if chr(message[0]) == 'd':
                    print(f"[Python] Received connect message '{connectMsg[:40]}...'")
                    self.connect_json = connectMsg
                    break

            else:
                print(f"[Python] Unrecognized message '{chr(message[0])}'; expected 'c' or 'd'")
                continue

            self._mq_res.send(b"c")

        self._mq_res.send(b"d") # Done

    def __del__(self):
        if self._shm_inp is not None:
            self._shm_inp.detach()
            self._shm_inp = None
        if self._shm_res is not None:
            self._shm_res.detach()
            self._shm_res = None

    def events(self):
        print("[Python] TriggerDataSource.events() called")

        while True:
            message, priority = self._mq_inp.receive()
            #print(f"[Python] Received msg '{message}', prio '{priority}'")

            if chr(message[0]) == 'g':
                event = Event(self._shm_inp, self._shm_inp_bufSizes)
                yield event
            elif chr(message[0]) == 's':
                break
            else:
                print(f"[Python] Unrecognized message '{chr(message[0])}' received")

    def result(self, persist, monitor):
        view = memoryview(self._shm_res)
        result = rdg.ResultDgram(view, persist, monitor)

        self._mq_res.send(b"g")

        #print(
        #    f"[Python] Sent message 'g'"
        #)


# Revisit: Move this into a .pyx?
class Event(object):
    def __init__(self, shm_inp, shm_bufSizes):
        self._shm_inp      = shm_inp
        self._shm_bufSizes = shm_bufSizes
        self._idx = 0
        self._pid = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx == len(self._shm_bufSizes) - 1:
            raise StopIteration

        view = memoryview(self._shm_inp)
        beg = self._shm_bufSizes[self._idx]
        end = self._shm_bufSizes[self._idx + 1]
        datagram = edg.EbDgram(view=view[beg:end])
        if datagram.pulseId() == 0:
            raise StopIteration

        self._idx += 1

        # Consistency check to make sure we haven't run off the end
        if self._pid == 0:
            self._pid = datagram.pulseId()
        elif datagram.pulseId() != self._pid:
            print(f"[Python] PulseId mismatch: "
                  f"expected {'%014x'%self._pid}, "
                  f"got {'%014lx'%datagram.pulseId()}")
            raise StopIteration

        return datagram
