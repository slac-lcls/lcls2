from psdaq.cas.config_scan_base import ConfigScanBase
import numpy as np
import json
import logging

nAsics = 4
nColumns = 384


def main():

    # default command line arguments
    defargs = {
        "--events": 1000,
        "--hutch": "tst",
        "--detname": "epixuhr3x2",
        "--scantype": "chargeinj",
        "--record": 0,
        "--config": "BEAM",
        "--nprocs": 2,
        "--run_type": "CHARGE_INJ",
    }

    aargs = [
        (
            "--gain-modes",
            {
                "type": str,
                "default": "FHG, FMG, FLG1, FLG2, AHLG1, AHLG2, AMLG1, AMLG2",
                "help": (
                    "Gain modes to use (default ['FHG ', "
                    "'FMG ', 'FLG1 ', 'FLG2 ', 'AHLG1 ', "
                    "'AHLG2 ', 'AMLG1 ', 'AMLG2 '])"
                ),
            },
        ),
        (
            "--skip_x",
            {
                "type": int,
                "default": 0,
                "help": "Number of pixels to skip along the X dimension.",
            },
        ),
        (
            "--skip_y",
            {
                "type": int,
                "default": 0,
                "help": "Number of pixels to skip along the Y dimension.",
            },
        ),
    ]
    scan = ConfigScanBase(userargs=aargs, defargs=defargs)

    args = scan.args

    gains_dict = {}

    gains_dict["FHG"] = {"level": 36, "stop": 6400, "start": 0, "step": 1}
    gains_dict["FMG"] = {"level": 44, "stop": 10000, "start": 0, "step": 1}
    gains_dict["FLG1"] = {"level": 5, "stop": 28000, "start": 0, "step": 1}
    gains_dict["FLG2"] = {"level": 37, "stop": 28000, "start": 0, "step": 1}
    gains_dict["AHLG1"] = {"level": 20, "stop": 28000, "start": 0, "step": 1}
    gains_dict["AHLG2"] = {"level": 52, "stop": 28000, "start": 0, "step": 1}
    gains_dict["AMLG1"] = {"level": 28, "stop": 28000, "start": 0, "step": 1}
    gains_dict["AMLG2"] = {"level": 60, "stop": 28000, "start": 0, "step": 1}

    keys = []

    for i in range(args.nprocs):
        keys.append(f"{args.detname}_{i}:user.Gain.SetSameGain4All")
        keys.append(f"{args.detname}_{i}:user.Gain.UsePixelMap")
        keys.append(f"{args.detname}_{i}:user.Gain.SetGainValue")
        keys.append(f"{args.detname}_{i}:user.Gain.UsePixelMap")
        keys.append(f"{args.detname}_{i}:user.Gain.PixelBitMapSel")

        keys.append(f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.enable")
        keys.append(f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.dacEn")
        keys.append(f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.rampEn")
        keys.append(f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.SKIP_X")
        keys.append(f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.SKIP_Y")
        keys.append(f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.dacStartValue")
        keys.append(f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.dacStopValue")
        keys.append(f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.dacStepValue")

        keys.append(f"{args.detname}_{i}:expert.FebFpga.App.WaveformControl.InjEn")

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

        if "," in args.gain_modes:
            gain_modes = args.gain_modes.split(",")
        else:
            gain_modes = [args.gain_modes]

        step = 0

        for i in range(args.nprocs):
            d[f"{args.detname}_{i}:user.Gain.SetSameGain4All"] = 1

            use_pix_map: int = 0 if args.skip_x == 0 and args.skip_y == 0 else 1
            d[f"{args.detname}_{i}:user.Gain.UsePixelMap"] = use_pix_map
            d[f"{args.detname}_{i}:user.Gain.PixelBitMapSel"] = 7

            d[f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.enable"] = 1
            d[f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.dacEn"] = 1
            d[f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.rampEn"] = 1
            d[f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.SKIP_X"] = args.skip_x
            d[f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.SKIP_Y"] = args.skip_y

            d[f"{args.detname}_{i}:expert.FebFpga.App.WaveformControl.InjEn"] = 1

        for gain_mode in gain_modes:
            gain_mode = gain_mode.strip()
            events = int(
                (gains_dict[gain_mode]["stop"]) / gains_dict[gain_mode]["step"]
            )
            pulserStart = gains_dict[gain_mode]["start"]
            pulserStop = gains_dict[gain_mode]["stop"]
            pulserStep = gains_dict[gain_mode]["step"]

            print(f"gain mode: {gain_mode}")
            print(f"Start:{pulserStart}, Stop:{pulserStop}, Step:{pulserStep}")
            print(gains_dict[gain_mode]["level"])

            metad["gain_mode"] = gain_mode
            metad["step"] = step
            metad["events"] = events

            for i in range(args.nprocs):
                d[f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.dacStartValue"] = int(
                    pulserStart
                )
                d[f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.dacStopValue"] = int(
                    pulserStop
                )
                d[f"{args.detname}_{i}:user.FebFpga.App.VINJ_DAC.dacStepValue"] = int(
                    pulserStep
                )

                d[f"{args.detname}_{i}:user.Gain.SetGainValue"] = gains_dict[gain_mode][
                    "level"
                ]

            yield (d, float(step), json.dumps(metad))
            step += 1

    scan.run(keys, steps)


if __name__ == "__main__":
    main()

