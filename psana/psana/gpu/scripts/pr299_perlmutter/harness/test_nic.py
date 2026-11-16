"""Mock-only NIC evidence tests; never touch real NICs or Weka data."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nic


class NICTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.net = self.root / "net"
        self.net.mkdir()
        self.ib = self.root / "ib"
        self.ib.mkdir()
        for name, value in (("NET_ROOT", self.net), ("IB_ROOT", self.ib)):
            patcher = mock.patch.object(nic, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.command = mock.patch.object(nic, "_command", side_effect=self.command_output).start()
        self.addCleanup(mock.patch.stopall)
        self.interface("eth0")

    def interface(self, name, pci="0000:01:00.0", port=0):
        device = self.root / pci
        device.mkdir(exist_ok=True)
        interface = self.net / name
        interface.mkdir()
        (interface / "device").symlink_to(device)
        (interface / "dev_port").write_text(str(port))
        (interface / "statistics").mkdir()
        (interface / "statistics/rx_bytes").write_text("999999")
        (interface / "statistics/tx_bytes").write_text("888888")
        return interface

    def command_output(self, *args):
        if args[0] == "findmnt":
            return json.dumps({"filesystems": [{"target": "/weka", "source": "weka",
                                                "fstype": "wekafs", "options": "rw"}]})
        if args[:2] == ("ethtool", "-i"):
            return "driver: mlx5_core\nversion: 1\nbus-info: 0000:01:00.0\n"
        return "NIC statistics:\n rx_bytes_phy: 100\n tx_bytes_phy: 40\n rx_packets: 3\n"

    def samples(self):
        before = nic.snapshot(nic.inventory())
        before["timestamp"] = 1.0
        after = copy.deepcopy(before)
        after["timestamp"] = 3.0
        for values in after["hardware"].values():
            values["rx_bytes"] += 200
            values["tx_bytes"] += 80
        return before, after

    def test_inventory_aliases_and_separate_optional_counters(self):
        self.interface("alias0")
        self.interface("eth1", port=1)
        self.interface("vf0", pci="0000:02:00.0")
        (self.root / "0000:02:00.0/physfn").symlink_to(self.root / "0000:01:00.0")
        (self.net / "lo").mkdir()
        counters = self.ib / "mlx5_0/ports/1/counters"
        counters.mkdir(parents=True)
        (self.ib / "mlx5_0/device").symlink_to(self.root / "0000:01:00.0")
        (counters / "port_rcv_data").write_text("12")
        (counters / "port_xmit_data").write_text("7")
        result = nic.inventory()
        json.dumps(result, allow_nan=False)
        self.assertEqual(len(result["hardware"]), 2)
        self.assertEqual(result["hardware"]["0000:01:00.0/port0"]["aliases"], ["alias0", "eth0"])
        self.assertEqual(set(result["interfaces"]), {"alias0", "eth0", "eth1"})
        self.assertEqual(result["initial"]["ib"]["mlx5_0/port1"], {"rx_bytes": 48, "tx_bytes": 28})
        self.assertEqual(result["initial"]["netdev"]["eth0"]["rx_bytes"], 999999)
        self.assertEqual(result["interfaces"]["eth0"]["ethtool_i"]["driver"], "mlx5_core")
        self.command.assert_any_call("findmnt", "-J", "-T", str(nic.DATA_ROOT), "-o",
                                     "TARGET,SOURCE,FSTYPE,OPTIONS")

    def test_non_weka_and_malformed_mount_fail(self):
        for output in ('{"filesystems":[{"fstype":"nfs","source":"weka"}]}', "bad", '{}'):
            with self.subTest(output=output):
                self.command.side_effect = None
                self.command.return_value = output
                with self.assertRaises(nic.NICError):
                    nic.inventory()

    def test_missing_or_invalid_hardware_is_fatal(self):
        for stats in ("rx_bytes_phy: 5", "rx_bytes_phy: -1\ntx_bytes_phy: 3",
                      "rx_bytes_phy: bad\ntx_bytes_phy: 3"):
            with self.subTest(stats=stats):
                self.command.side_effect = lambda *args: stats if args[1] == "-S" else self.command_output(*args)
                with self.assertRaises(nic.NICError):
                    nic.inventory()

    def test_no_ports_and_missing_driver_fail(self):
        with mock.patch.object(nic, "_identity", return_value=None):
            with self.assertRaises(nic.NICError):
                nic.inventory()
        self.command.side_effect = lambda *args: "version: 1" if args[1] == "-i" else self.command_output(*args)
        with self.assertRaisesRegex(nic.NICError, "No physical hardware"):
            nic.inventory()

    def test_command_timeout_and_missing_binary_fail(self):
        with mock.patch.object(nic, "_command", new=ORIGINAL_COMMAND):
            for error in (FileNotFoundError("ethtool"), nic.subprocess.TimeoutExpired("ethtool", 10),
                          nic.subprocess.CalledProcessError(1, "ethtool")):
                with mock.patch.object(nic.subprocess, "run", side_effect=error):
                    with self.assertRaises(nic.NICError):
                        nic._command("ethtool", "-S", "eth0")

    def test_optional_errors_do_not_replace_hardware(self):
        inventory = nic.inventory()
        (self.net / "eth0/statistics/rx_bytes").unlink()
        result = nic.snapshot(inventory)
        self.assertEqual(result["netdev"], {})
        self.assertIn("eth0", result["optional_errors"]["netdev"])
        self.assertEqual(result["hardware"]["0000:01:00.0/port0"]["rx_bytes"], 100)

    def test_snapshot_rejects_changed_node_mapping_and_duplicates(self):
        original = nic.inventory()
        for change in ("node", "mapping", "duplicate"):
            with self.subTest(change=change):
                inventory = copy.deepcopy(original)
                if change == "node":
                    inventory["hostname"] += "-other"
                elif change == "mapping":
                    inventory["hardware"]["0000:01:00.0/port0"]["interface"] = "missing"
                else:
                    inventory["hardware"]["alias"] = inventory["hardware"]["0000:01:00.0/port0"]
                with self.assertRaises(nic.NICError):
                    nic.snapshot(inventory)

    def test_delta_hardware_only(self):
        before, after = self.samples()
        after["netdev"]["eth0"]["rx_bytes"] = 0
        result = nic.delta(before, after)
        self.assertEqual(result, {"elapsed_s": 2.0, "hardware": {"0000:01:00.0/port0": {
            "rx_bytes": 200, "tx_bytes": 80, "rx_Bps": 100.0, "tx_Bps": 40.0}}})

    def test_delta_rejects_bad_timestamps_keys_counters(self):
        before, valid = self.samples()
        for change in ("reset", "negative", "missing", "ports", "node", "float", "bool",
                       "zero_time", "backward_time", "nan", "inf"):
            with self.subTest(change=change):
                after = copy.deepcopy(valid)
                values = after["hardware"]["0000:01:00.0/port0"]
                if change in ("reset", "negative", "float", "bool"):
                    values["rx_bytes"] = {"reset": 99, "negative": -1, "float": 301.0, "bool": True}[change]
                elif change == "missing":
                    del values["tx_bytes"]
                elif change == "ports":
                    after["hardware"] = {}
                elif change == "node":
                    after["hostname"] += "-other"
                else:
                    after["timestamp"] = {"zero_time": 1, "backward_time": 0,
                                          "nan": float("nan"), "inf": float("inf")}[change]
                with self.assertRaises(nic.NICError):
                    nic.delta(before, after)

    def test_monitor_first_last_and_maxima(self):
        first, last = self.samples()
        middle = copy.deepcopy(last)
        middle["timestamp"] = 2.0
        last["timestamp"] = 4.0
        with mock.patch.object(nic, "snapshot", side_effect=[first, middle, last]), \
                mock.patch.object(nic.threading, "Thread"):
            monitor = nic.Monitor({}, self.root / "samples.jsonl")
            self.assertIs(monitor.start(), monitor)
            monitor._record()
            summary = monitor.stop()
        lines = [json.loads(line) for line in (self.root / "samples.jsonl").read_text().splitlines()]
        self.assertEqual(lines, [first, middle, last])
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["hardware"], nic.delta(first, last)["hardware"])
        self.assertEqual(summary["max_sampled_rx_Bps"], {"0000:01:00.0/port0": 200.0})
        self.assertTrue(monitor._stream.closed)
        with self.assertRaises(nic.NICError):
            monitor.start()
        with self.assertRaises(nic.NICError):
            monitor.stop()

    def test_monitor_real_worker_error_propagates(self):
        first, second = self.samples()
        second["hardware"]["0000:01:00.0/port0"]["rx_bytes"] = 0
        for failure in (OSError("lost stats"), second):
            with self.subTest(failure=failure):
                with mock.patch.object(nic, "snapshot", side_effect=[first, failure]):
                    monitor = nic.Monitor({}, self.root / ("worker-{}.jsonl".format(type(failure).__name__)), 0.001)
                    monitor.start()
                    monitor._thread.join(timeout=2)
                    self.assertFalse(monitor._thread.is_alive())
                    with self.assertRaisesRegex(nic.NICError, "worker failed"):
                        monitor.stop()
                    self.assertTrue(monitor._stream.closed)

    def test_monitor_final_failure_and_interval_validation(self):
        first, unused = self.samples()
        for interval in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                nic.Monitor({}, self.root / "invalid", interval)
        with mock.patch.object(nic, "snapshot", side_effect=[first, nic.NICError("missing stats")]), \
                mock.patch.object(nic.threading, "Thread"):
            monitor = nic.Monitor({}, self.root / "final.jsonl").start()
            with self.assertRaisesRegex(nic.NICError, "missing stats"):
                monitor.stop()
            self.assertTrue(monitor._stream.closed)

    def test_monitor_start_failure_and_existing_output(self):
        output = self.root / "start.jsonl"
        with mock.patch.object(nic, "snapshot", side_effect=nic.NICError("missing stats")):
            monitor = nic.Monitor({}, output)
            with self.assertRaisesRegex(nic.NICError, "missing stats"):
                monitor.start()
            self.assertTrue(monitor._stream.closed)
        output.write_text("preserve evidence\n")
        with mock.patch.object(nic, "snapshot") as snapshot:
            with self.assertRaises(FileExistsError):
                nic.Monitor({}, output).start()
            snapshot.assert_not_called()
        self.assertEqual(output.read_text(), "preserve evidence\n")

    def test_probe_bounded_read_and_cached_failure(self):
        first, last = self.samples()
        idle = copy.deepcopy(first)
        idle["timestamp"] = 2.0
        source = self.root / "run51.xtc2"
        source.write_bytes(b"abcdefghijk")
        for active in (True, False):
            output = self.root / "probe-{}.json".format(active)
            ending = copy.deepcopy(last if active else idle)
            ending["timestamp"] = 3.0
            with mock.patch.object(nic, "PROBE_FILE", source), \
                    mock.patch.object(nic, "PROBE_BYTES", 8), \
                    mock.patch.object(nic, "snapshot", side_effect=[first, idle, ending]), \
                    mock.patch.object(nic.Path, "open", mock.mock_open(read_data=b"abcdefghijk")) as read_file, \
                    mock.patch.object(nic.time, "sleep") as sleep:
                if active:
                    result = nic.probe({}, output)
                    self.assertEqual(result["bytes_read"], 8)
                else:
                    with self.assertRaisesRegex(nic.NICError, "lacked substantial hardware RX.*cached"):
                        nic.probe({}, output)
                sleep.assert_called_once_with(1.0)
                read_file().read.assert_called_once_with(8)
            result = json.loads(output.read_text())
            self.assertEqual(result["hardware_rx_active"], active)
            self.assertEqual(source.read_bytes(), b"abcdefghijk")
            self.assertEqual(result["baseline"]["hardware"]["0000:01:00.0/port0"]["rx_bytes"], 0)
            self.assertIn("not exact Weka", result["attribution"])

    def test_probe_rejects_short_file_before_sampling(self):
        source = self.root / "short.xtc2"
        source.write_bytes(b"short")
        output = self.root / "short.json"
        with mock.patch.object(nic, "PROBE_FILE", source), \
                mock.patch.object(nic, "snapshot") as snapshot:
            with self.assertRaisesRegex(nic.NICError, "shorter"):
                nic.probe({}, output)
            snapshot.assert_not_called()
        self.assertFalse(output.exists())


ORIGINAL_COMMAND = nic._command

if __name__ == "__main__":
    unittest.main()
