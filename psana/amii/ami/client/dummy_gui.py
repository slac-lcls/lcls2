import zmq
from ami.operation.array import ROI
from ami.manager.graph import Graph

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://localhost:5555")                                                                               
sock.send_string('get_config')                                                                               
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://localhost:5555")                                                                               
sock.send_string('get_config')                                                                               

graph = Graph(sock.recv_pyobj())

box = ROI.make_box()

# add/change/delete/connect/disconnect methods? (disconnect for the edge)
graph.add(box, 'roi2', [('src','array')])
print graph.graph_dict

# send modified graph back to master

print 'hello'
