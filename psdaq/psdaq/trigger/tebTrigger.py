import sys
import numpy
import posix_ipc
import mmap
import logging
import json
import argparse

from struct import unpack
import EbDgram     as edg
import ResultDgram as rdg
import CubeResultDgram as qdg


class ArgsParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__()
        self.add_argument('-p', type=int, choices=range(0, 8), default=0, help='partition (default 0)')
        self.add_argument('-b', type=str, required=True, help='IPC key base value')

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
        key_base = self.args.b[1:]

        try:
            self._mq_inp = posix_ipc.MessageQueue("/mqtebinp_" + key_base, read=True, write=False)
        except posix_ipc.Error as exp:
            print(
                f"[Python] Error connecting to 'Inputs' message queue - Error: {exp}"
            )
            sys.exit(1)

        try:
            self._mq_res = posix_ipc.MessageQueue("/mqtebres_" + key_base, read=False, write=True)
        except posix_ipc.Error as exp:
            print(
                f"[Python] Error connecting to 'Results' message queue - Error: {exp}"
            )
            sys.exit(1)

        print(f"[Python] Connected to message queues")

        while True: # Synch up when there's cruft in the pipe
            message, priority = self._mq_inp.receive()
            print(f"[Python] Received message '{message}', prio '{priority}'")

            if chr(message[0]) != 'i':
                print(f"[Python] Unrecognized message '{chr(message[0])}'; expected 'i'")
            else:
                try:
                    shm_msg = message.decode().split(',')
                    self._shm_inp = posix_ipc.SharedMemory(shm_msg[1], size=int(shm_msg[2]))
                    inputsSize = 0
                    self._shm_inp_bufSizes = []
                    for ctrbSize in shm_msg[3:]:
                        self._shm_inp_bufSizes.append(inputsSize)
                        inputsSize += int(ctrbSize)
                    self._shm_inp_bufSizes.append(inputsSize)
                    self._shm_inp_mmap = mmap.mmap(self._shm_inp.fd, self._shm_inp.size)

                except posix_ipc.Error as exp:
                    print(
                        f"[Python] Error connecting to 'Inputs' shared memory - Error: {exp}"
                    )
                    sys.exit(1)

                print(f"[Python] Set up Inputs shared memory key {shm_msg[1]}")
                break

        self._mq_res.send(b"g")

        while True: # Synch up when there's cruft in the pipe
            message, priority = self._mq_inp.receive()
            print(f"[Python] Received message '{message}', prio '{priority}'")

            if chr(message[0]) != 'r':
                print(f"[Python] Unrecognized message '{chr(message[0])}'; expected 'r'")
            else:
                try:
                    shm_msg = message.decode().split(',')
                    self._shm_res = posix_ipc.SharedMemory(shm_msg[1], size=int(shm_msg[2]))
                    self._shm_res_mmap = mmap.mmap(self._shm_res.fd, self._shm_res.size)
                except posix_ipc.Error as exp:
                    print(
                        f"[Python] Error connecting to 'Results' shared memory - Error: {exp}"
                    )
                    self._shm_inp.unlink()
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
                connectMsg += self._shm_inp_mmap[0:size].decode()
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
            self._shm_inp.unlink()
            self._shm_inp = None
        if self._shm_res is not None:
            self._shm_res.unlink()
            self._shm_res = None

    def events(self):
        print("[Python] TriggerDataSource.events() called")

        while True:
            message, priority = self._mq_inp.receive()
            #print(f"[Python] Received msg '{message}', prio '{priority}'")

            if chr(message[0]) == 'g':
                event = Event(self._shm_inp_mmap, self._shm_inp_bufSizes, int(message[1:],16))
                yield event
            elif chr(message[0]) == 's':
                break
            else:
                print(f"[Python] Unrecognized message '{chr(message[0])}' received")

    def result(self, persist, monitor):

        result = rdg.ResultDgram(self._shm_res_mmap, persist, monitor)
        self._mq_res.send(b"g")

        #print(
        #    f"[Python] Sent message 'g'"
        #)

    def cubeResult(self, persist, monitor, bin_index, worker, bin_record):
#
#  Can't seem to make Cython use the CubeResultDgram/ResultDgram inheritance.
#
#        result = qdg.CubeResultDgram(self._shm_res_mmap, persist, monitor, 
#                                     bin_index, worker, bin_record)
#
#        logging.warning(f'cubeResult bin {bin_index} wrk {worker}  data {result.data()}')

        result = rdg.ResultDgram(self._shm_res_mmap, persist, monitor)
        aux =  ((bin_index&0x3ff)<<0) | ((worker&0x3f)<<10) | ((bin_record&1)<<16) 
        result.auxdata(aux)

#        logging.warning(f'cubeResult bin {bin_index} wrk {worker} aux {aux:x} data {result.data():x}')

        self._mq_res.send(b"g")

# Revisit: Move this into a .pyx?
class Event(object):
    def __init__(self, shm_inp_mmap, shm_bufSizes, ctrb):
        self._shm_inp_mmap      = shm_inp_mmap
        self._shm_bufSizes = shm_bufSizes
        self._ctrb = ctrb
        self._idx = 0
        self._pid = None

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._idx == len(self._shm_bufSizes):
                raise StopIteration
            if (self._ctrb >> self._idx)&1:
                break
            self._idx += 1

        beg = self._shm_bufSizes[self._idx]
        end = self._shm_bufSizes[self._idx + 1]
        datagram = edg.EbDgram(view=self._shm_inp_mmap[beg:end])

        self._idx += 1

        # Consistency check to make sure we haven't run off the end
        if self._pid is None:
            self._pid = datagram.pulseId()
        elif datagram.pulseId() != self._pid:
            print(f"[Python] PulseId mismatch: "
                  f"expected {'%014x'%self._pid}, "
                  f"got {'%014x'%datagram.pulseId()}, src {datagram.xtc.src.value():x}")
            raise StopIteration

        return datagram
