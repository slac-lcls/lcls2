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
host, id, flags, env, cmd = ('host', 'id', 'flags', 'env', 'cmd')
task_set = ''
"""
    o_file.writelines(header + "\n")

    with open(cnf_file, "r") as f:
        for line in f:
            if line.find("platform:") > -1 or line.find("taskset") > -1:
                continue
            o_file.writelines(line)
    o_file.close()
    print(f"Done. The cnf file {cnf_file} has been converted to {py_file}")


def _do_main():
    typer.run(main)


if __name__ == "__main__":
    _do_main()
