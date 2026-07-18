#!/usr/bin/env python3
"""
net_bandwidth.py

Sample per-interface network byte counters on the current node and print
receive/transmit bandwidth. This is useful for checking node-level read
bandwidth while running psana jobs against FFB/Weka, Perlmutter Lustre, or
other network storage.

Example usage:

    # Sample all non-virtual interfaces once per second for 60 seconds.
    python psana/psana/debugtools/net_bandwidth.py --samples 60

    # Run on a Slurm allocation node.
    srun -w sdfampere027 -n 1 python psana/psana/debugtools/net_bandwidth.py \
        --interval 1 --samples 120

    # Only show likely high-speed interfaces.
    python psana/psana/debugtools/net_bandwidth.py --include 'ib*' --include 'en*'

    # Measure InfiniBand/RDMA port counters. The IB port_data counters are
    # reported in 4-byte words and are converted to B/s here.
    python psana/psana/debugtools/net_bandwidth.py --source ib --samples 60

    # Measure NIC physical counters from ethtool -S. This is useful on Weka
    # clients where RDMA/storage traffic may not appear in netdev rx_bytes.
    python psana/psana/debugtools/net_bandwidth.py --source ethtool --samples 60

    # Measure Perlmutter Slingshot/CXI physical octets. Lustre uses KFI/CXI and
    # therefore does not appear in the Linux hsn* netdev rx_bytes counters.
    python psana/psana/debugtools/net_bandwidth.py --source cxi --samples 60

    # Restrict CXI to optimized Slingshot traffic instead of all good octets.
    python psana/psana/debugtools/net_bandwidth.py --source cxi \
        --cxi-rx-counter hni_sts_rx_octets_opt \
        --cxi-tx-counter hni_sts_tx_octets_opt --samples 60

    # Machine-readable output.
    python psana/psana/debugtools/net_bandwidth.py --jsonl --samples 10
"""

import argparse
import fnmatch
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_EXCLUDE = (
    "lo",
    "docker*",
    "veth*",
    "virbr*",
    "br-*",
    "cni*",
    "flannel*",
    "kube*",
    "podman*",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure node network receive/transmit bandwidth from netdev, "
            "InfiniBand, CXI telemetry, or NIC hardware counters."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Notes:
  - Rates are bytes/second by default, matching Grafana's net bytes panel.
  - recv is the useful first proxy for FFB/Weka read bandwidth.
  - Use --list to inspect interface names before choosing --include.
  - Use --samples 0 to run until Ctrl-C.
""",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between counter samples (default: 1.0)",
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=10,
        help="Number of rate samples to print; 0 means run until Ctrl-C (default: 10)",
    )
    parser.add_argument(
        "--source",
        choices=("netdev", "ib", "cxi", "ethtool", "all"),
        default="netdev",
        help=(
            "Counter source: netdev uses /sys/class/net rx/tx byte counters; "
            "ib uses /sys/class/infiniband port_rcv_data/port_xmit_data "
            "converted from 4-byte words to bytes; cxi uses Slingshot hardware "
            "octet counters from /sys/class/cxi and includes Perlmutter Lustre "
            "KFI traffic; ethtool uses NIC hardware counters from ethtool -S; "
            "all uses all sources and may double count traffic "
            "(default: netdev)."
        ),
    )
    parser.add_argument(
        "--cxi-rx-counter",
        default="hni_sts_rx_ok_octets",
        help=(
            "CXI telemetry receive counter to use with --source cxi "
            "(default: hni_sts_rx_ok_octets)"
        ),
    )
    parser.add_argument(
        "--cxi-tx-counter",
        default="hni_sts_tx_ok_octets",
        help=(
            "CXI telemetry transmit counter to use with --source cxi "
            "(default: hni_sts_tx_ok_octets)"
        ),
    )
    parser.add_argument(
        "--ethtool-rx-counter",
        default="rx_bytes_phy",
        help="ethtool -S receive counter to use with --source ethtool (default: rx_bytes_phy)",
    )
    parser.add_argument(
        "--ethtool-tx-counter",
        default="tx_bytes_phy",
        help="ethtool -S transmit counter to use with --source ethtool (default: tx_bytes_phy)",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="Interface glob to include, e.g. 'ib*' or 'en*'. Can be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "Interface glob to exclude. Defaults exclude common virtual "
            "interfaces; repeated values add to that list."
        ),
    )
    parser.add_argument(
        "--include-down",
        action="store_true",
        help="Include interfaces whose operstate is down.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected interfaces and exit.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print total recv/sent rates, not per-interface rates.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Print one JSON record per sample.",
    )
    parser.add_argument(
        "--bytes",
        action="store_true",
        help="Print raw B/s numbers instead of human-readable SI units.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not print the human-readable table header.",
    )
    return parser.parse_args()


def read_text(path, default=""):
    try:
        return path.read_text().strip()
    except OSError:
        return default


def read_int(path, default=0):
    try:
        return int(read_text(path, str(default)))
    except ValueError:
        return default


def read_counter(path):
    """Read a plain integer or a CXI telemetry value of form VALUE@TIMESTAMP."""
    value = path.read_text().strip().split("@", 1)[0]
    return int(value)


def is_selected(name, operstate, args):
    includes = args.include
    excludes = DEFAULT_EXCLUDE + tuple(args.exclude)

    if includes and not any(fnmatch.fnmatch(name, pat) for pat in includes):
        return False
    if any(fnmatch.fnmatch(name, pat) for pat in excludes):
        return False
    if not args.include_down and operstate == "down":
        return False
    return True


def interface_info(args):
    infos = []
    if args.source in ("netdev", "all"):
        infos.extend(netdev_info(args))
    if args.source in ("ib", "all"):
        infos.extend(ib_info(args))
    if args.source in ("cxi", "all"):
        infos.extend(cxi_info(args))
    if args.source in ("ethtool", "all"):
        infos.extend(ethtool_info(args))
    return infos


def netdev_info(args):
    infos = []
    for iface in sorted(Path("/sys/class/net").iterdir()):
        name = iface.name
        operstate = read_text(iface / "operstate", "unknown")
        if not is_selected(name, operstate, args):
            continue
        speed_mbit = read_int(iface / "speed", -1)
        infos.append(
            {
                "name": name,
                "source": "netdev",
                "label": name,
                "operstate": operstate,
                "speed_mbit": speed_mbit,
                "rx_path": iface / "statistics" / "rx_bytes",
                "tx_path": iface / "statistics" / "tx_bytes",
                "scale": 1,
            }
        )
    return infos


def ib_info(args):
    infos = []
    for counter_dir in sorted(Path("/sys/class/infiniband").glob("*/ports/*/counters")):
        device = counter_dir.parents[2].name
        port = counter_dir.parent.name
        label = f"{device}/port{port}"
        if args.include and not any(fnmatch.fnmatch(label, pat) for pat in args.include):
            continue
        if any(fnmatch.fnmatch(label, pat) for pat in args.exclude):
            continue
        rx_path = counter_dir / "port_rcv_data"
        tx_path = counter_dir / "port_xmit_data"
        if not rx_path.exists() or not tx_path.exists():
            continue
        infos.append(
            {
                "name": label,
                "source": "ib",
                "label": label,
                "operstate": "unknown",
                "speed_mbit": -1,
                "rx_path": rx_path,
                "tx_path": tx_path,
                "scale": 4,
            }
        )
    return infos


def cxi_info(args):
    """Discover Slingshot/CXI telemetry counters, including KFI traffic."""
    infos = []
    for cxi_path in sorted(Path("/sys/class/cxi").glob("cxi*")):
        telemetry = cxi_path / "device" / "telemetry"
        rx_path = telemetry / args.cxi_rx_counter
        tx_path = telemetry / args.cxi_tx_counter
        if not rx_path.exists() or not tx_path.exists():
            continue

        net_paths = sorted((cxi_path / "device" / "net").glob("*"))
        if net_paths:
            net_path = net_paths[0]
            label = net_path.name
            operstate = read_text(net_path / "operstate", "unknown")
            speed_mbit = read_int(net_path / "speed", -1)
        else:
            label = cxi_path.name
            operstate = "unknown"
            speed_mbit = -1

        if not is_selected(label, operstate, args):
            continue
        infos.append(
            {
                "name": f"cxi:{label}",
                "source": "cxi",
                "label": label,
                "operstate": operstate,
                "speed_mbit": speed_mbit,
                "rx_path": rx_path,
                "tx_path": tx_path,
                "scale": 1,
            }
        )
    return infos


def ethtool_info(args):
    infos = []
    for iface in sorted(Path("/sys/class/net").iterdir()):
        name = iface.name
        operstate = read_text(iface / "operstate", "unknown")
        if not is_selected(name, operstate, args):
            continue
        counters = read_ethtool_counters(name)
        if (
            args.ethtool_rx_counter not in counters
            or args.ethtool_tx_counter not in counters
        ):
            continue
        infos.append(
            {
                "name": f"ethtool:{name}",
                "source": "ethtool",
                "label": name,
                "operstate": operstate,
                "speed_mbit": read_int(iface / "speed", -1),
                "iface": name,
                "rx_counter": args.ethtool_rx_counter,
                "tx_counter": args.ethtool_tx_counter,
                "scale": 1,
            }
        )
    return infos


def read_ethtool_counters(iface):
    try:
        out = subprocess.check_output(
            ["ethtool", "-S", iface],
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}

    counters = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            counters[key] = int(value)
        except ValueError:
            continue
    return counters


def sample(infos):
    values = {}
    for info in infos:
        name = info["name"]
        try:
            if info["source"] == "ethtool":
                counters = read_ethtool_counters(info["iface"])
                rx = counters[info["rx_counter"]]
                tx = counters[info["tx_counter"]]
            else:
                rx = read_counter(info["rx_path"])
                tx = read_counter(info["tx_path"])
        except (OSError, ValueError):
            continue
        except KeyError:
            continue
        values[name] = (rx, tx)
    return values


def fmt_rate(value, raw_bytes=False):
    if raw_bytes:
        return f"{value:.0f} B/s"
    units = ("B/s", "kB/s", "MB/s", "GB/s", "TB/s")
    rate = float(value)
    for unit in units:
        if abs(rate) < 1000.0 or unit == units[-1]:
            return f"{rate:7.2f} {unit}"
        rate /= 1000.0
    return f"{rate:.2f} B/s"


def print_interface_list(infos):
    host = socket.gethostname()
    print(f"host={host}")
    print(f"{'source':8s} {'interface':20s} {'state':10s} {'speed'}")
    for info in infos:
        speed = info["speed_mbit"]
        speed_text = "unknown" if speed < 0 else f"{speed} Mbit/s"
        print(
            f"{info['source']:8s} {info['label']:20s} "
            f"{info['operstate']:10s} {speed_text}"
        )


def print_table_header(summary_only):
    if summary_only:
        print(f"{'time':19s} {'host':20s} {'recv_total':>15s} {'sent_total':>15s}")
    else:
        print(
            f"{'time':19s} {'host':20s} {'interface':20s} "
            f"{'recv':>15s} {'sent':>15s}"
        )


def emit_json(host, timestamp, dt, rates, totals):
    print(
        json.dumps(
            {
                "time": timestamp,
                "host": host,
                "interval_s": dt,
                "interfaces": rates,
                "total_recv_Bps": totals[0],
                "total_sent_Bps": totals[1],
            },
            sort_keys=True,
        )
    )


def emit_table(host, timestamp, rates, totals, args):
    if args.summary_only:
        print(
            f"{timestamp:19s} {host:20s} "
            f"{fmt_rate(totals[0], args.bytes):>15s} "
            f"{fmt_rate(totals[1], args.bytes):>15s}"
        )
        return

    for name in sorted(rates):
        rx, tx = rates[name]
        print(
            f"{timestamp:19s} {host:20s} {name:20s} "
            f"{fmt_rate(rx, args.bytes):>15s} "
            f"{fmt_rate(tx, args.bytes):>15s}"
        )
    print(
        f"{timestamp:19s} {host:20s} {'TOTAL':20s} "
        f"{fmt_rate(totals[0], args.bytes):>15s} "
        f"{fmt_rate(totals[1], args.bytes):>15s}"
    )


def main():
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    if args.samples < 0:
        raise ValueError("--samples must be >= 0")

    infos = interface_info(args)
    if args.list:
        print_interface_list(infos)
        return
    if not infos:
        print("No selected network interfaces.", file=sys.stderr)
        sys.exit(2)

    host = socket.gethostname()
    prev = sample(infos)
    prev_t = time.monotonic()

    if not args.jsonl and not args.no_header:
        print_table_header(args.summary_only)

    count = 0
    try:
        while args.samples == 0 or count < args.samples:
            time.sleep(args.interval)
            now = time.monotonic()
            cur = sample(infos)
            dt = now - prev_t
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            rates = {}
            total_rx = 0.0
            total_tx = 0.0
            info_by_name = {info["name"]: info for info in infos}
            for name, info in info_by_name.items():
                if name not in prev or name not in cur:
                    continue
                scale = info["scale"]
                rx = max(0, cur[name][0] - prev[name][0]) * scale / dt
                tx = max(0, cur[name][1] - prev[name][1]) * scale / dt
                rates[name] = {
                    "recv_Bps": rx,
                    "sent_Bps": tx,
                }
                total_rx += rx
                total_tx += tx

            if args.jsonl:
                emit_json(host, timestamp, dt, rates, (total_rx, total_tx))
            else:
                table_rates = {
                    name: (rate["recv_Bps"], rate["sent_Bps"])
                    for name, rate in rates.items()
                }
                emit_table(host, timestamp, table_rates, (total_rx, total_tx), args)

            prev = cur
            prev_t = now
            count += 1
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
