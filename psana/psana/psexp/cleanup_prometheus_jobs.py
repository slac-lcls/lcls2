from psana.psexp import PrometheusManager
import typer
from typing_extensions import Annotated
import os


def main(
    exp: Annotated[str, typer.Argument(help="Experiment code")],
    runnum: Annotated[int, typer.Argument(help="Run no.")],
    user: Annotated[str, typer.Argument(help="User who ran the job")],
    n_ranks: Annotated[int, typer.Argument(help="No. of total mpi ranks")],
):
    job = f"{exp}_{runnum}_{user}"
    prom_man = PrometheusManager(exp, runnum, job=job)
    prom_man.delete_all_metrics_on_pushgateway(n_ranks=n_ranks)


def _do_main():
    typer.run(main)


if __name__ == "__main__":
    _do_main()
