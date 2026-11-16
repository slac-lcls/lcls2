"""Strict node-level NIC sampling; hardware, netdev and IB are never added together."""

import json
import math
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

DATA_ROOT = Path("/sdf/data/lcls/ds/prj/public01/xtc")
PROBE_FILE = DATA_ROOT / "mfx100848724-r0051-s000-c000.xtc2"
PROBE_BYTES = 128 * 1024 * 1024
NET_ROOT = Path("/sys/class/net")
IB_ROOT = Path("/sys/class/infiniband")
ATTRIBUTION = "Node-level physical-port traffic, not exact Weka-only bytes."


class NICError(RuntimeError):
    """Required hardware evidence is missing or inconsistent."""


def _command(*args):
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True,
                              timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise NICError("{} failed: {}".format(" ".join(args), error)) from error


def _fields(text):
    return {key.strip(): value.strip() for key, separator, value in
            (line.partition(":") for line in text.splitlines()) if separator}


def _weka_mount(path):
    try:
        mount = json.loads(_command("findmnt", "-J", "-T", str(path), "-o",
                                    "TARGET,SOURCE,FSTYPE,OPTIONS"))["filesystems"]
        if len(mount) != 1 or mount[0]["fstype"].lower() not in {
                "wekafs", "weka", "fuse.weka", "fuse.wekafs"}:
            raise NICError("{} is not a verified Weka mount: {}".format(path, mount))
        return {"path": str(path), "findmnt": mount[0]}
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise NICError("Invalid findmnt evidence for {}".format(path)) from error


def _text(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _identity(interface):
    device = (interface / "device").resolve()
    pci = device.name
    if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", pci):
        return None
    if (device / "physfn").exists():
        return None
    port = _text(interface / "dev_port")
    if not port:
        port = _text(interface / "dev_id")
    try:
        port = int(port, 16 if port.startswith("0x") else 10)
        if port < 0:
            raise ValueError(port)
    except ValueError as error:
        raise NICError("Missing/invalid physical port for {}".format(interface)) from error
    return pci.lower(), port


def _counts(rx, tx, scale=1):
    try:
        values = {"rx_bytes": int(rx) * scale, "tx_bytes": int(tx) * scale}
        if min(values.values()) < 0:
            raise ValueError("negative counter")
        return values
    except (ValueError, TypeError) as error:
        raise NICError("Invalid NIC counters: {!r}, {!r}".format(rx, tx)) from error


def _hardware(interface):
    values = _fields(_command("ethtool", "-S", interface))
    try:
        return _counts(values["rx_bytes_phy"], values["tx_bytes_phy"])
    except KeyError as error:
        raise NICError("{} missing required rx_bytes_phy/tx_bytes_phy".format(interface)) from error


def _optional(specs):
    values, errors = {}, {}
    for name, spec in specs.items():
        try:
            values[name] = _counts(Path(spec["rx_path"]).read_text(),
                                   Path(spec["tx_path"]).read_text(), spec["scale"])
        except (OSError, NICError) as error:
            errors[name] = str(error)
    return values, errors


def inventory():
    """Return JSON-safe mount, PCI/port aliases, driver info and initial counters.

    At least one PCI physical interface must expose both hardware counters.
    Unsupported ports are inventoried but never replace required counters.
    """
    result = {"hostname": socket.gethostname(), "mount": _weka_mount(DATA_ROOT),
              "interfaces": {}, "hardware": {}, "netdev": {}, "ib": {}, "unsupported": {}}
    for interface in sorted(NET_ROOT.iterdir()):
        identity = _identity(interface)
        if identity is None:
            continue
        pci, port = identity
        name, key = interface.name, "{}/port{}".format(pci, port)
        try:
            driver = _fields(_command("ethtool", "-i", name))
            if not driver.get("driver"):
                raise NICError("{} missing ethtool -i driver information".format(name))
            hardware = _hardware(name)
        except NICError as error:
            result["unsupported"][name] = {"pci": pci, "port": port, "error": str(error)}
            continue
        result["interfaces"][name] = {
            "pci": pci, "port": port, "ethtool_i": driver,
            "phys_port_name": _text(interface / "phys_port_name"),
            "operstate": _text(interface / "operstate"), "hardware": hardware}
        spec = result["hardware"].setdefault(
            key, {"interface": name, "pci": pci, "port": port, "aliases": []})
        spec["aliases"].append(name)
        result["netdev"][name] = {
            "rx_path": str(interface / "statistics/rx_bytes"),
            "tx_path": str(interface / "statistics/tx_bytes"), "scale": 1}
    for counters in sorted(IB_ROOT.glob("*/ports/*/counters")):
        device, port = counters.parents[2], counters.parent.name
        result["ib"]["{}/port{}".format(device.name, port)] = {
            "pci": (device / "device").resolve().name, "port": port,
            "rx_path": str(counters / "port_rcv_data"),
            "tx_path": str(counters / "port_xmit_data"), "scale": 4}
    result["initial"] = snapshot(result)
    return result


def snapshot(inventory):
    """Return monotonic timestamp and hardware bytes, with separate optional data."""
    if not inventory["hardware"] or inventory["hostname"] != socket.gethostname():
        raise NICError("No physical hardware ports, or inventory belongs to another node")
    result = {"hostname": inventory["hostname"], "hardware": {}, "optional_errors": {}}
    identities = set()
    for key, spec in inventory["hardware"].items():
        identity = (spec["pci"], spec["port"])
        if identity in identities or key != "{}/port{}".format(*identity):
            raise NICError("Duplicate/inconsistent physical port: {}".format(key))
        identities.add(identity)
        if _identity(NET_ROOT / spec["interface"]) != identity:
            raise NICError("Physical mapping changed for {}".format(spec["interface"]))
        result["hardware"][key] = _hardware(spec["interface"])
    for source in ("netdev", "ib"):
        result[source], result["optional_errors"][source] = _optional(inventory[source])
    result["timestamp"] = time.monotonic()
    return result


def delta(before, after):
    """Hardware-only {elapsed_s, hardware: {PCI/port: bytes and rx/tx_Bps}}."""
    elapsed = after["timestamp"] - before["timestamp"]
    if (not all(math.isfinite(sample["timestamp"]) for sample in (before, after))
            or not math.isfinite(elapsed) or elapsed <= 0):
        raise NICError("Sample timestamps must be finite and strictly increasing")
    if (before["hostname"] != after["hostname"] or not before["hardware"]
            or before["hardware"].keys() != after["hardware"].keys()):
        raise NICError("Hardware port keys or node changed between samples")
    result = {"elapsed_s": elapsed, "hardware": {}}
    for key, current in after["hardware"].items():
        previous = before["hardware"][key]
        if set(current) != {"rx_bytes", "tx_bytes"} or set(previous) != set(current):
            raise NICError("Missing hardware statistics for {}".format(key))
        rates = {}
        for direction in ("rx", "tx"):
            field = direction + "_bytes"
            if any(type(sample[field]) is not int or sample[field] < 0
                   for sample in (previous, current)) or current[field] < previous[field]:
                raise NICError("Invalid/reset counter: {} {}".format(key, field))
            rates[field] = current[field] - previous[field]
            rates[direction + "_Bps"] = rates[field] / elapsed
        result["hardware"][key] = rates
    return result


class Monitor:
    """Single-use sampler. start() is synchronous; stop() joins and rethrows errors.

    JSONL contains snapshots (including first and final). Output must not exist.
    stop() returns first-to-last delta plus per-port maximum interval RX B/s.
    """

    def __init__(self, inventory, output_path, interval=1.0):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be finite and positive")
        self.inventory, self.output_path, self.interval = inventory, str(output_path), interval
        self._event, self._thread, self._error = threading.Event(), None, None
        self._first, self._last, self._count, self._max_rx = None, None, 0, {}
        self._stream, self._started, self._stopped = None, False, False

    def _record(self):
        current = snapshot(self.inventory)
        if self._last is not None:
            for key, values in delta(self._last, current)["hardware"].items():
                self._max_rx[key] = max(self._max_rx.get(key, 0), values["rx_Bps"])
        self._stream.write(json.dumps(current, allow_nan=False) + "\n")
        self._stream.flush()
        if self._first is None:
            self._first = current
        self._last, self._count = current, self._count + 1

    def _run(self):
        try:
            while not self._event.wait(self.interval):
                self._record()
        except Exception as error:
            self._error = error

    def start(self):
        if self._started:
            raise NICError("Monitor is single-use")
        self._started = True
        self._stream = open(self.output_path, "x", encoding="utf-8")
        try:
            self._record()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            self._stream.close()
            raise
        return self

    def stop(self):
        if self._thread is None or self._stopped:
            raise NICError("Monitor is not running")
        self._event.set()
        self._thread.join()
        try:
            if self._error is not None:
                raise NICError("NIC monitor worker failed: {}".format(self._error)) from self._error
            self._record()
            return dict(delta(self._first, self._last), samples=self._count,
                        max_sampled_rx_Bps=dict(self._max_rx), output_path=self.output_path,
                        attribution=ATTRIBUTION)
        finally:
            self._stream.close()
            self._stopped = True


def probe(inventory, output_path):
    """Read 128 MiB after 1 s idle; require substantial physical RX above idle.

    Ordinary sequential reads only: cache hits are possible, no cache eviction.
    Background traffic can still account for RX; subtraction is only an estimate.
    Output must not exist. No output is success evidence if this function raises.
    """
    path = PROBE_FILE.resolve(strict=True)
    mount = _weka_mount(path)
    if path.stat().st_size < PROBE_BYTES:
        raise NICError("Probe file is shorter than {} bytes: {}".format(PROBE_BYTES, path))
    with open(output_path, "x", encoding="utf-8") as output:
        idle_start = snapshot(inventory)
        time.sleep(1.0)
        read_start = snapshot(inventory)
        baseline = delta(idle_start, read_start)
        remaining = PROBE_BYTES
        with path.open("rb", buffering=0) as source:
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise NICError("Unexpected EOF during Weka probe")
                remaining -= len(chunk)
        read_end = snapshot(inventory)
        traffic = delta(read_start, read_end)
        excess = {port: max(0, values["rx_bytes"] - baseline["hardware"][port]["rx_Bps"] * traffic["elapsed_s"])
                  for port, values in traffic["hardware"].items()}
        active = any(value >= PROBE_BYTES / 4 for value in excess.values())
        result = {"file": str(path), "bytes_read": PROBE_BYTES, "mount": mount,
                  "baseline": baseline, "during_read": traffic, "hardware_rx_active": active,
                  "rx_above_idle_estimate": excess, "required_single_port_excess_bytes": PROBE_BYTES // 4,
                  "attribution": ATTRIBUTION, "samples": [idle_start, read_start, read_end]}
        json.dump(result, output, allow_nan=False)
        output.write("\n")
    if not active:
        raise NICError("Weka probe lacked substantial hardware RX above idle; reads may be cached. Evidence: {}".format(output_path))
    return result
