######################################################
# Run zmq.PULL socket that keeps listening to db info 
# (run_no, node_name, port_no)and display psplot when 
# new info arrive.
######################################################
from psana.psexp.zmq_utils import SubSocket
import socket
import zmq
import subprocess


pull_socket = None


def get_db_info():
    global pull_socket
    # We acquire an available port using a socket then closing it immediately 
    # so that this port can be used later.
    if not pull_socket:
        IPAddr = socket.gethostbyname(socket.gethostname())
        sock = socket.socket()
        sock.bind(('', 0))
        port_no = sock.getsockname()[1]
        sock.close()
        supervisor_ip_addr = f'{IPAddr}:{port_no}'
        socket_name = f"tcp://{supervisor_ip_addr}"
        print(f'Run psana with this socket: {socket_name}', flush=True)
        pull_socket = SubSocket(socket_name, socket_type=zmq.PULL)
    # Fetch data (received data are a tubple of compressed and uncompressed dict)
    obj = pull_socket.recv_zipped_pickle()[1]
    return obj 


def start_psplot():
    p = subprocess.Popen(["tail", "-f", "~/.bashrc"])
    print(f"Process ID of subprocess {p.pid}")
    # Send SIGTER 
    p.terminate()
    # Wait for process to terminate
    returncode = p.wait()
    print(f"Returncode of subprocess: {returncode}")


if __name__ == "__main__":
    runnum, node, port = (0, None, None)
    while True:
        obj = get_db_info()
        if obj['node'] != node or obj['port'] != port or obj['runnum'] > runnum:
            runnum, node, port = (obj['runnum'], obj['node'], obj['port'])
            print(f'Received new {runnum=} {node=} {port=}', flush=True)
            start_psplot()
        else:
            print(f'Received old {obj}')





