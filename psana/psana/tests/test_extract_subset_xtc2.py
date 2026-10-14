import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "extract_subset_xtc2.py"


def load_extract_subset_module():
    spec = importlib.util.spec_from_file_location("extract_subset_xtc2", str(APP_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_dgrams(module, path):
    total = 0
    events = 0
    with open(path, "rb") as infile:
        while True:
            datagram = module.read_datagram(infile)
            if datagram is None:
                break
            total += 1
            _raw_bytes, service, _timestamp = datagram
            if service in module.EVENT_TRANSITIONS:
                events += 1
    return total, events


def test_parse_event_indices():
    module = load_extract_subset_module()
    assert module.parse_event_indices("0,2,4-6") == set([0, 2, 4, 5, 6])


def test_parse_detector_names_accepts_comma_and_space_separated_inputs():
    module = load_extract_subset_module()
    assert module.parse_detector_names(["jungfrau,feespec"]) == ["jungfrau", "feespec"]
    assert module.parse_detector_names(["jungfrau", "feespec,pcav"]) == [
        "jungfrau",
        "feespec",
        "pcav",
    ]


def test_copy_filtered_xtc_file(tmp_path):
    module = load_extract_subset_module()
    input_path = Path(__file__).resolve().parent / "test_data" / "intg_det" / "xpptut15-r0014-s000-c000.xtc2"
    output_path = tmp_path / input_path.name

    input_total, input_events = count_dgrams(module, input_path)
    result = module.copy_filtered_xtc_file(
        str(input_path),
        str(output_path),
        event_indices=set([0, 2, 4]),
    )
    output_total, output_events = count_dgrams(module, output_path)

    assert result["written_events"] == 3
    assert result["total_events"] == input_events
    assert output_events == 3
    assert output_total == input_total - input_events + 3


def test_extract_selected_streams_creates_smalldata_layout(tmp_path):
    module = load_extract_subset_module()
    test_data_dir = Path(__file__).resolve().parent / "test_data" / "intg_det"
    xtc_file = test_data_dir / "xpptut15-r0014-s000-c000.xtc2"
    smd_file = test_data_dir / "smalldata" / "xpptut15-r0014-s000-c000.smd.xtc2"

    results = module.extract_selected_streams(
        xtc_files=[str(xtc_file)],
        smd_files=[str(smd_file)],
        output_dir=str(tmp_path),
        num_events=2,
    )

    xtc_output = tmp_path / xtc_file.name
    smd_output = tmp_path / "smalldata" / smd_file.name

    assert xtc_output.exists()
    assert smd_output.exists()
    assert results["xtc"][0]["written_events"] == 2
    assert results["smd"][0]["written_events"] == 2


def test_copy_filtered_xtc_file_by_timestamp(tmp_path):
    module = load_extract_subset_module()
    input_path = Path(__file__).resolve().parent / "test_data" / "intg_det" / "xpptut15-r0014-s001-c000.xtc2"
    output_path = tmp_path / input_path.name

    selected_timestamps = []
    with open(input_path, "rb") as infile:
        while len(selected_timestamps) < 2:
            datagram = module.read_datagram(infile)
            if datagram is None:
                break
            _raw_bytes, service, timestamp = datagram
            if service in module.EVENT_TRANSITIONS:
                selected_timestamps.append(timestamp)

    result = module.copy_filtered_xtc_file(
        str(input_path),
        str(output_path),
        selected_timestamps=set(selected_timestamps),
    )

    assert result["written_events"] == 2


def test_index_selection_callback_keeps_global_count_across_invocations():
    module = load_extract_subset_module()
    callback = module._make_index_selection_callback(event_indices=set([3, 7]))

    class FakeEvent(object):
        def __init__(self, timestamp):
            self.timestamp = timestamp

    class FakeRun(object):
        def __init__(self, timestamps):
            self._timestamps = timestamps

        def events(self):
            for ts in self._timestamps:
                yield FakeEvent(ts)

    batch1 = list(callback(FakeRun([10, 11, 12, 13, 14])))
    batch2 = list(callback(FakeRun([15, 16, 17, 18, 19])))

    assert [evt.timestamp for evt in batch1] == [13]
    assert [evt.timestamp for evt in batch2] == [17]
    assert callback.found_indices == set([3, 7])


@pytest.mark.skipif(shutil.which("smdwriter") is None, reason="smdwriter is required")
def test_generated_smalldata_can_be_read_back(tmp_path):
    module = load_extract_subset_module()
    test_data_dir = Path(__file__).resolve().parent / "test_data" / "intg_det"
    xtc_file = test_data_dir / "xpptut15-r0014-s000-c000.xtc2"
    smd_file = test_data_dir / "smalldata" / "xpptut15-r0014-s000-c000.smd.xtc2"

    module.extract_selected_streams(
        xtc_files=[str(xtc_file)],
        smd_files=[str(smd_file)],
        output_dir=str(tmp_path),
        num_events=2,
    )

    script = (
        "from psana import DataSource\n"
        "ds = DataSource(exp='xpptut15', run=14, dir=r'%s', batch_size=1)\n"
        "run = next(ds.runs())\n"
        "count = 0\n"
        "for evt in run.events():\n"
        "    count += 1\n"
        "print(count)\n"
    ) % str(tmp_path)
    result = subprocess.run(
        ["python", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "2"
