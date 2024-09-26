from psana.psexp import PrometheusManager
import typer
from typing_extensions import Annotated
import os


def main(
    job_id: Annotated[
        str,
        typer.Argument(
            help="Prometheus JobID (default is slurm jobid or exp_runnum_user)"
        ),
    ],
    n_ranks: Annotated[int, typer.Argument(help="No. of total mpi ranks")],
):
    prom_man = PrometheusManager(job=job_id)
    prom_man.delete_all_metrics_on_pushgateway(n_ranks=n_ranks)


def _do_main():
    typer.run(main)


if __name__ == "__main__":
    _do_main()
