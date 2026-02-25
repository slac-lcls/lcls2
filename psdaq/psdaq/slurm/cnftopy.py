import typer
from typing_extensions import Annotated
from psdaq.procmgr.ProcMgr import deduce_platform


def _get_platform(in_file):
    # Prefer explicit platform in source file when available.
    with open(in_file, "r") as f:
        for line in f:
            s = line.strip()
            if s.startswith("platform ="):
                parts = s.split("=", 1)
                if len(parts) == 2:
                    return parts[1].strip().strip("'\"")
            if s.startswith("if not platform:") and "platform" in s and "=" in s:
                parts = s.split("=", 1)
                if len(parts) == 2:
                    return parts[1].strip().strip("'\"")
    try:
        return deduce_platform(in_file)
    except Exception:
        return "0"


def _skip_header_line(line):
    stripped = line.strip()
    return (
        stripped.startswith("platform:")
        or stripped.startswith("platform =")
        or stripped.startswith("if not platform:")
        or stripped == "import os"
        or stripped.startswith("CONDA_PREFIX =")
        or stripped.startswith("CONFIGDIR =")
        or stripped.startswith("host, cores, id, flags, env")
        or stripped.startswith("task_set = ''")
        or "taskset" in stripped
    )


def main(
    cnf_file: Annotated[
        str, typer.Argument(help="Input config file (.cnf or .py)")
    ],
    py_file: Annotated[str, typer.Argument(help="Output filename with .py extension")],
):
    platform = _get_platform(cnf_file)
    header = """
import os
CONDA_PREFIX = os.environ.get('CONDA_PREFIX','')
CONFIGDIR = '/cds/home/m/monarin/lcls2/psdaq/psdaq/slurm'
host, cores, id, flags, env, env_group, cmd, rtprio = ('host', 'cores', 'id', 'flags', 'env', 'env_group', 'cmd', 'rtprio')
task_set = ''
"""
    with open(py_file, "w") as o_file:
        o_file.writelines(f"platform = '{platform}'\n")
        o_file.writelines(header + "\n")

        in_ami_block = False
        in_ami_dict = False
        ami_dict_has_env_group = False
        with open(cnf_file, "r") as f:
            for line in f:
                if _skip_header_line(line):
                    continue
                if "procmgr_ami" in line and "=" in line:
                    in_ami_block = True
                if in_ami_block and line.strip().startswith("procmgr_config.extend(procmgr_ami)"):
                    in_ami_block = False

                if "id:'procstat'" in line or 'id:"procstat"' in line:
                    o_file.writelines(" {id: 'daqstat', cmd: 'daqstat -i 5'},\n")
                    continue
                if "cmd:drp_cmd" in line or "cmd:teb_cmd" in line:
                    # Add 10 cores as a default no. of physical cores need for drp and teb processes
                    cols = line.split(",")
                    cols.insert(1, "cores: 10")
                    line = ",".join(cols)

                if in_ami_block:
                    stripped = line.strip()
                    is_dict_start = stripped.startswith("{") or "append({" in stripped
                    is_dict_end = (
                        stripped == "}"
                        or stripped.endswith("},")
                        or stripped.endswith("})")
                    )

                    if is_dict_start:
                        in_ami_dict = True
                        ami_dict_has_env_group = "env_group" in line
                    elif in_ami_dict and "env_group" in line:
                        ami_dict_has_env_group = True

                    if in_ami_dict and "cmd:" in line and not ami_dict_has_env_group:
                        line = line.replace("cmd:", "env_group: 'ami', cmd:", 1)
                        ami_dict_has_env_group = True

                    if in_ami_dict and is_dict_end:
                        in_ami_dict = False
                        ami_dict_has_env_group = False

                o_file.writelines(line)
    print(f"Done. The file {cnf_file} has been converted to {py_file}")


def _do_main():
    typer.run(main)


if __name__ == "__main__":
    _do_main()
