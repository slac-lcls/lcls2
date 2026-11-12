import typer
from typing_extensions import Annotated
import subprocess
import logging
import os

logger = logging.getLogger(__name__)


def get_output_header(job_name, platform, nodelist, daq_cmd):
    envs = [
        "CONDA_PREFIX",
        "CONFIGDB_AUTH",
        "TESTRELDIR",
        "SUBMODULEDIR",
        "RDMAV_FORK_SAFE",
        "RDMAV_HUGEPAGES_SAFE",
        "OPENBLAS_NUM_THREADS",
        "PS_PARALLEL",
    ]
    env_dict = {key: os.environ.get(key, "none") for key in envs}

    job_id = int(os.environ.get("SLURM_JOB_ID", "-1"))
    # if job_id == -1:
    #    raise ValueError("No SLURM_JOB_ID found")

    header = ""
    header += f"# SLURM_JOB_ID:{job_id}\n"
    header += "# ID:      %s\n" % job_name
    header += "# PLATFORM:%s\n" % platform
    header += "# HOST:    %s\n" % nodelist
    header += "# CMDLINE: %s\n" % daq_cmd

    # obfuscating the password in the log
    clear_auth = env_dict["CONFIGDB_AUTH"]
    env_dict["CONFIGDB_AUTH"] = "*****"
    for env in envs:
        header += f"# {env}:{env_dict[env]}\n"
    env_dict["CONFIGDB_AUTH"] = clear_auth

    git_describe = None
    if "TESTRELDIR" in env_dict:
        try:
            git_output = subprocess.check_output(
                ["git", "-C", os.environ["TESTRELDIR"], "describe", "--dirty", "--tag"],
                stderr=subprocess.STDOUT,
                text=True
            ).strip()
            git_describe = git_output
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Git describe failed for TESTRELDIR '%s': %s",
                os.environ["TESTRELDIR"],
                e.output.strip()
            )

    if git_describe:
        header += "# GIT_DESCRIBE:%s\n" % git_describe

    print(header)


def main(
    job_name: Annotated[str, typer.Argument(help="Slurm job name")],
    platform: Annotated[int, typer.Argument(help="Platform number")],
    nodelist: Annotated[str, typer.Argument(help="List of nodes running this job")],
    daq_cmd: Annotated[str, typer.Argument(help="Full daq command")],
):
    get_output_header(job_name, platform, nodelist, daq_cmd)


def _do_main():
    typer.run(main)


if __name__ == "__main__":
    _do_main()
