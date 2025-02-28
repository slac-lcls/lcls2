import typer
from typing_extensions import Annotated
from psdaq.procmgr.ProcMgr import deduce_platform


def main(
    cnf_file: Annotated[str, typer.Argument(help="The original .cnf file")],
    py_file: Annotated[str, typer.Argument(help="Output filename with .py extension")],
):
    platform = deduce_platform(cnf_file)
    o_file = open(py_file, "w")
    o_file.writelines(f"platform = '{platform}'" + "\n")
    header = """
import os
CONDA_PREFIX = os.environ.get('CONDA_PREFIX','')
CONFIGDIR = '/cds/home/m/monarin/lcls2/psdaq/psdaq/slurm'
host, cores, id, flags, env, cmd, rtprio = ('host', 'cores', 'id', 'flags', 'env', 'cmd', 'rtprio')
task_set = ''
"""
    o_file.writelines(header + "\n")

    with open(cnf_file, "r") as f:
        for line in f:
            if line.find("platform:") > -1 or line.find("taskset") > -1:
                continue
            elif line.find("procstat") > -1:
                o_file.writelines(" {id: 'daqstat', cmd: 'daqstat -i 5'},")
            elif line.find("cmd:drp_cmd") > -1 or line.find("cmd:teb_cmd") > -1:
                # Add 10 cores as a default no. of physical cores need for drp and teb processes
                cols = line.split(",")
                cols.insert(1, "cores: 10")
                new_line = ",".join(cols)
                o_file.writelines(new_line)
            else:
                o_file.writelines(line)
    o_file.close()
    print(f"Done. The cnf file {cnf_file} has been converted to {py_file}")


def _do_main():
    typer.run(main)


if __name__ == "__main__":
    _do_main()
