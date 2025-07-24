#!/cds/home/m/monarin/lcls2/setup_env.sh python3

import subprocess
import os
import re
import shlex
from datetime import datetime
from psdaq.slurm.main import Runner
import smtplib
from email.message import EmailMessage

PYCFGS = [
    "/cds/home/opr/tmoopr/scripts/neh-base.py",
    "/cds/home/opr/tmoopr/scripts/hsd.py",
    "/cds/home/opr/rixopr/scripts/rix-hsd.py"
]

IGNORE_UIDS = [
    "hsdpvs_tmo_41_a",
    "hsdpvs_tmo_41_b",
    "hsdioc_tmo_41"
]

now = datetime.now()
LOGDIR = f"/cds/home/m/monarin/logs/{now:%Y/%m}"
LOGFILE = os.path.join(LOGDIR, f"process_check_{now:%Y%m%d_%H%M%S}.log")
os.makedirs(LOGDIR, exist_ok=True)

def get_entries_from_cfg(cfg_file):
    runner = Runner(cfg_file)
    entries = []
    for uid, detail in runner.config.items():
        host = detail.get("host", "unknown")
        cmd = detail.get("cmd", "")
        entries.append((host, uid, cmd))
    return entries

def get_pid_and_check_zombie(exe_name, host):
    pids = []
    zombies = []
    try:
        ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", host, f"pgrep -f {shlex.quote(exe_name)}"]
        out = subprocess.check_output(ssh_cmd, text=True)
        pids = [int(pid) for pid in out.splitlines()]
        for pid in pids:
            status_cmd = f"cat /proc/{pid}/status"
            check_cmd = ["ssh", host, status_cmd]
            try:
                status_output = subprocess.check_output(check_cmd, text=True)
                if any("State:\tZ" in line for line in status_output.splitlines()):
                    zombies.append(pid)
            except Exception as e:
                print(f'[ERROR] check status for {pid=} error {e=}')
                continue
    except Exception as e:
        print(f'[ERROR] ssh to {host=} error {e=}')
        pass
    return pids, zombies

def get_job_output_file(uid, user):
    try:
        out = subprocess.check_output(["squeue", "-u", user, "--format=%i %j", "-h"], text=True)
        for line in out.splitlines():
            if uid in line:
                jobid = line.split()[0]
                detail = subprocess.check_output(["scontrol", "show", "job", jobid], text=True)
                match = re.search(r"StdOut=(\S+)", detail)
                if match:
                    return match.group(1)
    except Exception as e:
        print(f'[ERROR] squeue {user=} error {e=}')
        pass
    return None

def log_has_error(logfile):
    try:
        with open(logfile) as f:
            for line in f:
                if any(x in line for x in ["Aborted", "Traceback", "General Error", "core dumped"]):
                    return True
    except Exception as e:
        print(f'[ERROR] logging {logfile=} error {e=}')
        pass
    return False

with open(LOGFILE, "w") as log:
    log.write(f"[INFO] Running check at {now}\n")

    for cfg in PYCFGS:
        user_match = re.search(r"/opr/(\w+?opr)/", cfg)
        user = user_match.group(1) if user_match else os.getlogin()
        log.write(f"[INFO] Checking cfg: {cfg}\n")
        entries = get_entries_from_cfg(cfg)

        for host, uid, cmd in entries:
            if uid in IGNORE_UIDS:
                log.write(f"[SKIP] Ignoring UID {uid}\n")
                continue
            exe_name = shlex.split(cmd)[0]
            log.write(f"[INFO] Checking {uid} on {host}: {exe_name}\n")
            pids, zombies = get_pid_and_check_zombie(exe_name, host)

            if not pids:
                log.write(f"[FAIL] No process found for {uid} (exe: {exe_name})\n")
                continue

            if zombies:
                for pid in zombies:
                    log.write(f"[FAIL] PID {pid} for {uid} is a zombie\n")
            else:
                logfile = get_job_output_file(uid, user)
                if logfile and log_has_error(logfile):
                    log.write(f"[FAIL] Found errors in log {logfile}\n")
                else:
                    log.write(f"[OK] {uid} is healthy\n")

print(f'Process check completed: {LOGFILE}')

# Send email if any FAIL lines are detected
with open(LOGFILE) as f:
    lines = f.readlines()

failures = [line for line in lines if "[FAIL]" in line]
if failures:
    msg = EmailMessage()
    msg["Subject"] = "Process Health Check: FAILURES detected"
    msg["From"] = "noreply@slac.stanford.edu"
    msg["To"] = "monarin@slac.stanford.edu"
    msg.set_content(
        "The following processes have issues:\n\n"
        + "".join(failures)
        + f"\n\nFull log: {LOGFILE}"
    )

    try:
        with smtplib.SMTP("localhost") as s:
            s.send_message(msg)
    except Exception as e:
        print(f"[WARN] Could not send email: {e}")
else:
    print("[INFO] All processes healthy, no email sent.")
