import os
import sys

import numpy as np
import pytest

from psana import DataSource, dgram
from psana.detector import DamageBitmask
from psana.dgramedit import AlgDef, DetectorDef, DgramEdit
from psana.psexp import TransitionId

BUFSIZE = 64000000


@pytest.fixture
def output_filename(tmp_path):
    fname = str(tmp_path / "out-dgramedit-test.xtc2")
    return fname


def create_array(dtype):
    if dtype in (np.float32, np.float64):
        arr = np.stack(
            [
                np.zeros(3, dtype=dtype) + np.finfo(dtype).min,
                np.zeros(3, dtype=dtype) + np.finfo(dtype).max,
            ]
        )
    else:
        arr = np.stack(
            [
                np.arange(np.iinfo(dtype).min, np.iinfo(dtype).min + 3, dtype=dtype),
                np.arange(
                    np.iinfo(dtype).max - 2, np.iinfo(dtype).max + 1, dtype=dtype
                ),
            ]
        )
    return arr


def check_output(fname):
    print(f"TEST OUTPUT by reading {fname} using DataSource")
    ds = DataSource(files=[fname])
    myrun = next(ds.runs())
    det = myrun.Detector("xpphsd")
    det2 = myrun.Detector("xppcspad")
    for evt in myrun.events():
        det.fex.show(evt)
        arrayRaw = det2.raw.raw(evt)
        knownArrayRaw = create_array(np.float32)
        # Convert the known array to the reformatted shape done by
        # the detecter interface.
        knownArrayRaw = np.reshape(knownArrayRaw, [1] + list(knownArrayRaw.shape))
        assert np.array_equal(arrayRaw, knownArrayRaw)
        print(f"det2 arrayRaw: {arrayRaw}")

        # Currently only checking one field from the second detector
        # assert np.array_equal(arrayRaw, create_array(np.float32))


@pytest.mark.skipif(sys.platform == "darwin", reason="check_output fails on macos")
def test_run_dgramedit(output_filename):
    run_dgramedit(output_filename)


def run_dgramedit(output_filename):
    ifname = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_data", "dgramedit-test.xtc2"
    )

    ds = DataSource(files=ifname)
    run = next(ds.runs())

    # Defines detector and alg.
    # Below example settings become hsd_fex_4_5_6 for its detector interface.
    algdef = AlgDef("fex", 4, 5, 6)
    detdef = DetectorDef("xpphsd", "hsd", "detnum1234")  # detname, dettype, detid

    # Define data formats
    datadef = {
        "valFex": (np.float32, 0),
        "strFex": (str, 1),
        "arrayFex0": (np.uint8, 2),
        "arrayFex1": (np.uint16, 2),
        "arrayFex2": (np.uint32, 2),
        "arrayFex3": (np.uint64, 2),
        "arrayFex4": (np.int8, 2),
        "arrayFex5": (np.int16, 2),
        "arrayFex6": (np.int32, 2),
        "arrayFex7": (np.int64, 2),
        "arrayFex8": (np.float32, 2),
        "arrayFex9": (np.float64, 2),
    }

    algdef2 = AlgDef("raw", 2, 3, 42)
    detdef2 = DetectorDef("xppcspad", "cspad", "detnum1234")
    datadef2 = {
        "arrayRaw": (np.float32, 2),
    }

    # Creates external output buffer for DgramEdit. Note that
    # a new dgram is always saved from the beginning of this buffer.
    xtc2buf = bytearray(BUFSIZE)

    # Writing out each dgram to the output file
    ofile = open(output_filename, "wb")

    # Allocating space for editable config
    config_buf = bytearray(BUFSIZE)
    config_buf[: run.configs[0]._size] = run.configs[0]
    config_editable = dgram.Dgram(view=memoryview(config_buf), offset=0)

    # Add new Names to config
    config_edit = DgramEdit(config_editable, bufsize=BUFSIZE)
    det = config_edit.Detector(detdef, algdef, datadef)
    det2 = config_edit.Detector(detdef2, algdef2, datadef2)
    config_edit.save(xtc2buf)
    ofile.write(xtc2buf[: config_edit.size])

    # Check if the new config is correct by looking inside
    # .software (detector names shown at root are those with ShapesData).
    new_config = config_edit.get_dgram()
    assert hasattr(new_config.software, "xpphsd")

    # Write out BeginRun using DgramEdit (no modification)
    beginrun_edit = DgramEdit(
        run._evt._dgrams[0], config_dgramedit=config_edit, bufsize=run._evt._dgrams[0]._size
    )
    beginrun_edit.save(xtc2buf)
    ofile.write(xtc2buf[: beginrun_edit.size])

    # Allocating space for editable L1Accept
    dgram_buf = bytearray(BUFSIZE)
    for i_evt, evt in enumerate(run.events()):
        dgram_buf[: evt._dgrams[0]._size] = evt._dgrams[0]
        dgram_editable = dgram.Dgram(
            config=run.configs[0], view=memoryview(dgram_buf), offset=0
        )
        dgram_edit = DgramEdit(
            dgram_editable, config_dgramedit=config_edit, bufsize=BUFSIZE
        )

        # Fill in data for previously given datadef (all fields
        # must be completed)
        det.fex.valFex = 1600.1234
        det.fex.strFex = "hello string"
        det.fex.arrayFex0 = create_array(np.uint8)
        det.fex.arrayFex1 = create_array(np.uint16)
        det.fex.arrayFex2 = create_array(np.uint32)
        det.fex.arrayFex3 = create_array(np.uint64)
        det.fex.arrayFex4 = create_array(np.int8)
        det.fex.arrayFex5 = create_array(np.int16)
        det.fex.arrayFex6 = create_array(np.int32)
        det.fex.arrayFex7 = create_array(np.int64)
        det.fex.arrayFex8 = create_array(np.float32)
        det.fex.arrayFex9 = create_array(np.float64)
        dgram_edit.adddata(det.fex)

        det2.raw.arrayRaw = create_array(np.float32)
        dgram_edit.adddata(det2.raw)
        if i_evt == 0:
            dgram_edit.updateservice(11)
            dgram_edit.updatedamage(1 << DamageBitmask.Corrupted.value)
            pydg = dgram_edit.get_pydgram()
            print(
                f"{i_evt=} {pydg.service()=} isEvent={TransitionId.isEvent(pydg.service())} damage={pydg.pyxtc.damage.value()}"
            )

        if i_evt == 4:
            dgram_edit.removedata("hsd", "raw")  # per event removal

        dgram_edit.save(xtc2buf)
        ofile.write(xtc2buf[: dgram_edit.size])

        # For non-configure dgram, you can retreive Dgram back by
        # passing in the output buffer where the Dgram was written to.
        new_dgram = dgram_edit.get_dgram(view=xtc2buf)
        if i_evt == 0:
            print(f"{i_evt=} {new_dgram.service()=}")
        assert new_dgram.xpphsd[0].fex.strFex == "hello string"

        if i_evt == 4:
            assert hasattr(new_dgram, "hsd") is False
            break

    # Other transitions FIXME: Mona makes other transitions
    # available through psana2 interface.
    # dgram = DgramEdit(pydg, config_dgramedit=config)
    # dgram.save(xtc2buf)
    # ofile.write(xtc2buf[:dgram.size])

    ofile.close()

    # Open the generated xtc2 file and test the value inside
    check_output(output_filename)


if __name__ == "__main__":
    test_run_dgramedit("test_output.xtc2")
