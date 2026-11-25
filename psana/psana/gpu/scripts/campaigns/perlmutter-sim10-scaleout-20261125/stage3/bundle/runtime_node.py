import socket,subprocess,sys
from pathlib import Path
subprocess.run([sys.executable,str(Path(__file__).with_name('runtime_check.py')),
                str(Path(sys.argv[1])/('runtime-C-'+socket.gethostname()+'.json'))],check=True,timeout=295)
