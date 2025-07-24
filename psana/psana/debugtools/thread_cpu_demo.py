import threading
import os

print(f"Start thread_cpu_demo pid: {os.getpid()}")
def get_cpu():
    import psutil
    return psutil.Process().cpu_num()

try:
    get_cpu = os.sched_getcpu  # Python 3.8+ only on Linux
except AttributeError:
    get_cpu = get_cpu()

def worker(thread_id):
    for _ in range(100):
        cpu = get_cpu()
        if _  == 0:
            print(f"Thread {thread_id:02d} running on CPU {cpu} pid: {os.getpid()}")

threads = []

for i in range(60):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()


