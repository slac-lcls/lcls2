######################################################
# Run zmq.PULL socket that keeps listening to db info 
# (run_no, node_name, port_no)and display psplot when 
# new info arrive.
######################################################
from psana.psexp.zmq_utils import PubSocket
import socket
import zmq
import subprocess
import sys


srv_socket = None
my_subproc = None


def get_db_info():
    global srv_socket
    # We acquire an available port using a socket then closing it immediately 
    # so that this port can be used later.
    if not srv_socket:
        IPAddr = socket.gethostbyname(socket.gethostname())
        sock = socket.socket()
        sock.bind(('', 0))
        port_no = sock.getsockname()[1]
        sock.close()
        supervisor_ip_addr = f'{IPAddr}:{port_no}'
        socket_name = f"tcp://{supervisor_ip_addr}"
        print(f'Run psana with this socket: {socket_name}', flush=True)
        srv_socket = PubSocket(socket_name, socket_type=zmq.PULL)
    # Fetch data 
    print(f'Getting data from client')
    obj = srv_socket.recv()
    return obj 


def start_psplot(node, port, detname):
    """ Start psplot as a subprocess with check to kill if
    there's an existing one."""
    global my_subproc
    if my_subproc is not None:
        # Send SIGTER 
        my_subproc.terminate()
        # Wait for process to terminate
        returncode = my_subproc.wait()
        print(f"Returncode of subprocess: {returncode}")
        my_subproc = None
    my_subproc = subprocess.Popen(["psplot", "-s", node, "-p", str(port), detname])
    print(f"Process ID of subprocess {my_subproc.pid}")


if __name__ == "__main__":
    runnum, node, port = (0, None, None)
    detname = sys.argv[1]
    while True:
        obj = get_db_info()
        if obj['node'] != node or obj['port'] != port or obj['runnum'] > runnum:
            runnum, node, port = (obj['runnum'], obj['node'], obj['port'])
            print(f'Received new {runnum=} {node=} {port=}', flush=True)
            start_psplot(node, port, detname)
        else:
            print(f'Received old {obj}')

