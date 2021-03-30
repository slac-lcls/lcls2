import socket
import struct
import sys
import numpy as np

assert(len(sys.argv)==2)

# from psdaq/drp/BldDetector.cc
server_address = ('', 10148) # BldDetectorSlow.cc says 12148 eventually
if sys.argv[1]=="-e": #ebeam
    multicast_group = '239.255.24.0'
    print('EBeam:')
elif sys.argv[1]=="-g": #gmd
    print('GMD:')
    multicast_group = '239.255.25.2'
elif sys.argv[1]=="-x": #xgmd
    print('XGMD:')
    multicast_group = '239.255.25.3'

# Create the socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind to the server address
sock.bind(server_address)

# Tell the operating system to add the socket to the multicast group
# on all interfaces.
group = socket.inet_aton(multicast_group)
mreq = struct.pack('4sL', group, socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

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

# Receive/respond loop
nevt = 0
while True:
    data, address = sock.recvfrom(32768)

    packet = BldPacket(np.frombuffer(data,dtype=np.uint32))
    if nevt==0: print('Received from addr/port',address[0],address[1])
    print('sec/nsec/fid/dmg:',hex(packet.seconds()),hex(packet.nanoseconds()),hex(packet.fiducials()),hex(packet.damage()))
    nevt+=1
    if nevt>3: break
