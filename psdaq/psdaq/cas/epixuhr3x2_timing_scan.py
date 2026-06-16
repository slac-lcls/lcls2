import json
import numpy as np
import numpy.typing as npt

from psdaq.cas.config_scan_base import ConfigScanBase


def main():

    # default command line arguments
    defargs = {
        "--events": 1000,
        "--hutch": "tst",
        "--detname": "epixuhr3x2",
        "--scantype": "timing",
        "--record": 0,
        "--config": "BEAM",
        "--nprocs": 2,
        "--run_type": "TIMING",
    }

    aargs = [
        (
            "--start_val",
            {
                "type": int,
                "default": 81000,
                "help": "Starting `start_ns` value to scan.",
            },
        ),
        (
            "--stop_val",
            {
                "type": int,
                "default": 115000,
                "help": "Final/stopping `start_ns` value to scan.",
            },
        ),
        (
            "--step_val",
            {"type": int, "default": 1000, "help": "Step value."},
        ),
    ]

    scan = ConfigScanBase(userargs=aargs, defargs=defargs)

    args = scan.args

    num_steps: int = int((args.stop_val - args.start_val) / args.step_val)

    scan_steps: npt.NDArray[np.uint32] = np.linspace(
        start=args.start_val, stop=args.stop_val, num=num_steps, dtype=np.uint32
    )

    keys = []

    for i in range(args.nprocs):
        keys.append(f"{args.detname}_{i}:user.start_ns")

    def steps():
        # The metad goes into the step_docstring of the timing DRP's BeginStep data
        # The step_docstring is used to guide the offline calibration routine
        metad = {
            "detname": args.detname,
            "scantype": args.scantype,
            "events": args.events,
        }
        d = {}
        step = 0

        for step, start_ns in enumerate(scan_steps):
            for i in range(args.nprocs):
                d[f"{args.detname}_{i}:user.start_ns"] = int(start_ns)

            print(f"start_ns: {start_ns}")

            metad["start_ns"] = int(start_ns)
            metad["step"] = step

            yield (d, int(step), json.dumps(metad))
            step += 1

    scan.run(keys, steps)


if __name__ == "__main__":
    main()

