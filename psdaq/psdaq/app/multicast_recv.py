import socket
import struct
import sys
import time
import numpy as np

assert(len(sys.argv)==2)

# can see these on the accelerator side on machine lcls-srv02
# from psdaq/drp/BldDetector.cc
server_address = ('', 10148) # BldDetectorSlow.cc says 12148 eventually
if sys.argv[1]=="-h": #HXR ebeam
    multicast_group = '239.255.24.0'
    print('EBeam:')
elif sys.argv[1]=="-s": #SXR ebeam
    multicast_group = '239.255.25.0'
    print('EBeam:')
elif sys.argv[1]=="-g": #gmd
    print('GMD:')
    multicast_group = '239.255.25.2'
elif sys.argv[1]=="-x": #xgmd
    print('XGMD:')
    multicast_group = '239.255.25.3'
elif sys.argv[1]=="-w": #xpp wave8
    print('XPP WAVE8:')
    multicast_group = '239.255.24.75'
elif sys.argv[1]=="-i": #xpp ipimb
    print('XPP IPIMB:')
    multicast_group = '239.255.24.40'
elif sys.argv[1]=="-p": #sxr pcav
    print('SXR PCAV:')
    multicast_group = '239.255.25.03'
elif sys.argv[1]=="-f": #fee spec
    print('FEE SPEC:')
    multicast_group = '239.255.24.46'

# Create the socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind to the server address
sock.bind(server_address)

# Tell the operating system to add the socket to the multicast group
# on all interfaces.
group = socket.inet_aton(multicast_group)
mreq = struct.pack('4sL', group, socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# matt has some c++ code to dump bld
# psdaq/psdaq/app/xcasttest.cc
class BldPacket:
    def __init__(self,array):
        self.array = array
    def fiducials(self):
        # fiducials also in the lower 17 bits of nanoseconds()
        return self.array[3]
    def seconds(self):
        # a guess from looking at the ebeam packet
        return self.array[1]
    def damage(self):
        # a guess from looking at the ebeam packet
        return self.array[5]
    def nanoseconds(self):
        # a guess from looking at the ebeam packet
        # lower 17 bits are the fiducial
        return self.array[0]
    def pulseId(self):
        # From the BLD DRP code
        return (self.array[3] << 32) + self.array[2]
    def bldId(self):
        # A guess from a comment in BldDetectorSlow.cc
        return self.array[4]

# Receive/respond loop
POSIX_TIME_AT_EPICS_EPOCH = 631152000
nevt = 0
wct_last = time.time_ns()
while True:
    data, address = sock.recvfrom(32768)

    packet = BldPacket(np.frombuffer(data,dtype=np.uint32))
    wct = time.time_ns()
    bt  = int((packet.seconds() + POSIX_TIME_AT_EPICS_EPOCH) * 1e9) + packet.nanoseconds()
    latency = (wct - bt) * 1e-9
    dt = (wct - wct_last) * 1e-9
    if nevt==0: print('Received from addr/port',address[0],address[1])
    if nevt<4:
        print('sec/nsec/pid/id/dmg:',hex(packet.seconds()),hex(packet.nanoseconds()),
                                     hex(packet.pulseId()),hex(packet.bldId()),hex(packet.damage()))
    if latency > 0.5 or dt > 5.0:
        print(f'{time.ctime()}: latency {latency}, wall clock {wct:x}, timestamp {bt:x}, recv gap {dt} s')
    wct_last = wct
    nevt+=1
    #if nevt>3: break
