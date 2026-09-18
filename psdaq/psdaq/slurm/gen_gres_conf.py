#!/usr/bin/env python3
"""Generate a Slurm gres.conf describing this node's GPU and datadev cards.

The GPU DRP pairs each datadev card with a GPU.  Both should ideally hang off the
same PCIe switch, so that the FPGA's peer-to-peer DMA into GPU memory does not
have to cross a root complex.  Measured on drp-srcf-gpu008, crossing costs nothing
at present -- every card is pinned at its own PCIe 4.0 x8 uplink (~102 Gbps), and
every hop beyond that has capacity to spare -- so a non-local pairing is reported
rather than refused.  That would change if the cards were ever x16 gen5.

The point of generating this rather than writing it by hand is determinism.  Slurm
otherwise allocates GPUs in bus order and happens to land on a good arrangement;
nothing guarantees that after a driver reload, a card swap or a Slurm upgrade.
And with 20-odd nodes coming, a hand-maintained mapping per node is a liability.

The Slurm controller (psslurmctld001) distributes slurm.conf and gres.conf to the
nodes from /var/spool/slurmd/conf-cache/ when it is restarted, so there is one global
gres.conf covering every node.  Hence every emitted line carries NodeName=: without it
Slurm would apply the line to all nodes, which cannot be right when each box has its
own PCI layout.

This prints one node's records and nothing else.  Publishing them is a deliberate
human step -- paste into psslurmctld001:/etc/slurm/gres.conf, replacing any existing
lines for the node, then 'sudo scontrol reconfigure' there.  That file is small enough
to read at a glance (two dozen lines for nine nodes), and editing it by hand is the
recovery path when something breaks at three in the morning, so nothing here writes to
it: a tool that rewrites a shared file has to be understood before it can be trusted,
and at 3 a.m. nobody has time to.

Run on the node, as any user (it reads /proc and /sys only), after every datadev
driver load:

    gen_gres_conf --expect N

then paste, then reconfigure.  The node's own line in slurm.conf needs a matching
Gres=, which this prints for you.  GresTypes already covers this: only the GPU is
declared as a gres, so 'GresTypes=gpu' suffices and needs no change.

Type names are the datadev's PCI bus number in hex, matching the device names the
driver produces with 'options datadev cfgDevName=1' (see cfgDevName in
aes-stream-drivers' common/driver/data_dev_top.c).  So a launch script that runs
with '-d /dev/datadev_84' requests '--gres=gpu:dd84:1' and the two identifiers cannot
drift apart.

The datadev itself is deliberately *not* declared as a gres, though doing so would let
Slurm catch a configuration that gave two processes the same card.  cgroup.conf here has
ConstrainDevices=yes, which per cgroup.conf(5) constrains a job's allowed devices to the
gres it was allocated -- so naming /dev/datadev_XX here would deny a CPU DRP access to
another lane of a card whose pairing a GPU DRP holds, and that arrangement is in use.
Catching duplicate -d arguments is better done by checking the cnf.

Installed as $TESTRELDIR/bin/gen_gres_conf by the [project.scripts] entry in
psdaq/pyproject.toml, which means build_all.sh has to be re-run to pick up an edit
made here.  While developing, run the file in place instead:

    python3 psdaq/psdaq/slurm/gen_gres_conf.py --check

It is deliberately dependency-free -- standard library only, no psdaq imports -- so
that running it in place works without a built tree, and so that --check can run
early in boot before anything else is up.
"""

import argparse
import datetime
import getpass
import glob
import os
import re
import shlex
import sys

PROC_DATADEV = "/proc/datadev_"
SYS_PCI      = "/sys/bus/pci/devices"
PROC_NVIDIA  = "/proc/driver/nvidia/gpus"


def read_datadevs():
    """Every datadev card, with its PCI address, firmware and GPU capability."""
    cards = []
    for entry in sorted(os.listdir("/proc")):
        if not entry.startswith("datadev_"):
            continue
        try:
            with open(os.path.join("/proc", entry)) as f:
                text = f.read()
        except OSError as exc:
            print(f"warning: cannot read /proc/{entry}: {exc}", file=sys.stderr)
            continue
        pci = re.search(r"^PCIe\[.*\]\s*:\s*(\S+)", text, re.M)
        fw  = re.search(r"^\s*Build String\s*:\s*([^:]+):", text, re.M)
        gpu = re.search(r"^\s*GPU Async En\s*:\s*(\d+)", text, re.M)
        # Easy to confuse with the above: 'GPU Async En' is the card's firmware, but
        # 'GPUAsync Support' is a plain #ifdef DATA_GPU, so it is the driver build.
        # A driver built without it never probes the register and reports every card
        # as incapable, whatever firmware they carry.
        drv = re.search(r"^\s*GPUAsync Support\s*:\s*(\S+)", text, re.M)
        if not pci:
            print(f"warning: no PCIe address in /proc/{entry}; skipping", file=sys.stderr)
            continue
        cards.append({"name":    entry,
                      "dev":     f"/dev/{entry}",
                      "pci":     pci.group(1).lower(),
                      "firmware": fw.group(1).strip() if fw else "unknown",
                      "gpu_capable": bool(gpu and int(gpu.group(1))),
                      "gpu_driver": bool(drv and drv.group(1).lower() == "enabled")})
    return cards


def read_gpus():
    """Every NVIDIA GPU, with its PCI address and char device minor."""
    gpus = []
    if not os.path.isdir(PROC_NVIDIA):
        return gpus
    for pci in sorted(os.listdir(PROC_NVIDIA)):
        try:
            with open(os.path.join(PROC_NVIDIA, pci, "information")) as f:
                text = f.read()
        except OSError as exc:
            print(f"warning: cannot read information for GPU {pci}: {exc}", file=sys.stderr)
            continue
        minor = re.search(r"^Device Minor:\s*(\d+)", text, re.M)
        excl  = re.search(r"^GPU Excluded:\s*(\S+)", text, re.M)
        bus   = re.search(r"^Bus Type:\s*(\S+)", text, re.M)
        if not minor:
            print(f"warning: no Device Minor for GPU {pci}; skipping", file=sys.stderr)
            continue

        # A GPU can be listed here and still be unusable.  When one fails, the driver asks
        # the kernel to remove it, and if anything holds it open the removal is refused --
        # "Attempting to remove device ... with non-zero usage count!" -- leaving an entry
        # that looks healthy.  /proc keeps listing it, so does lspci, and only nvidia-smi
        # (which we do not want to depend on) notices it is gone.  Pairing a card with such
        # a GPU produces a gres record Slurm allocates happily and a DRP that dies with
        # CUDA_ERROR_NO_DEVICE, which is exactly the invisible failure the pairing exists to
        # prevent.  Observed on drp-srcf-gpu008 after an Xid 154 GSP hang on 2026-09-17.
        #
        # Two independent symptoms, both read without opening the device: the driver reports
        # 'Bus Type: PCI' rather than 'PCIe' because it can no longer read the PCIe
        # capability structure, and current_link_speed returns EINVAL for the same reason.
        # Requiring both to agree avoids skipping a GPU over one odd reading.
        # Tested against "PCIe" rather than against the "PCI" that was observed, so that a
        # failure mode reporting something else again is also caught.  A missing Bus Type
        # line is treated as healthy: absence means the field moved or was renamed, which is
        # not evidence of a fault, and the link check below still applies.
        bus_type     = bus.group(1) if bus else None
        healthy_bus  = bus_type is None or bus_type == "PCIe"
        healthy_link = bool(pci_attr(pci.lower(), "current_link_speed"))
        if not healthy_bus and not healthy_link:
            print(f"warning: GPU {pci} is listed but not usable -- the driver reports "
                  f"'Bus Type: {bus_type}' rather than PCIe and cannot read its link speed, "
                  f"which means it has failed and its removal was refused (look for Xid in "
                  f"dmesg).  Not offered as a gres; a DRP given it would fail in CUDA init.",
                  file=sys.stderr)
            continue

        gpus.append({"pci":      pci.lower(),
                     "minor":    int(minor.group(1)),
                     "dev":      f"/dev/nvidia{int(minor.group(1))}",
                     "excluded": bool(excl and excl.group(1) != "No")})
    return gpus


def pci_path(pci):
    """The device's PCIe ancestry, root complex first.

    e.g. 0000:04:00.0 -> ['pci0000:00', '0000:00:01.1', '0000:01:00.0',
                          '0000:02:01.0']
    The immediate parent is the switch's *downstream port*, which every device has
    to itself, so proximity has to be judged by shared ancestry rather than by a
    common parent.
    """
    try:
        real = os.path.realpath(os.path.join(SYS_PCI, pci))
    except OSError:
        return []
    parts = real.split(os.sep)
    try:
        start = next(i for i, p in enumerate(parts) if p.startswith("pci0000:"))
    except StopIteration:
        return []
    return parts[start:-1]                  # drop the device itself


def shared_depth(lhs, rhs):
    """How many leading ancestors two devices have in common.

    0 means different root complexes.  1 means the same root complex but nothing
    below it.  2 or more means they share at least one bridge under the root
    complex, i.e. a PCIe switch, and can talk peer to peer without going up to the
    root complex.  That is what 'local' means below.
    """
    depth = 0
    for a, b in zip(lhs, rhs):
        if a != b:
            break
        depth += 1
    return depth


LOCAL_DEPTH = 2                             # Sharing a bridge below the root complex


def pci_attr(pci, attr):
    try:
        with open(os.path.join(SYS_PCI, pci, attr)) as f:
            return f.read().strip()
    except OSError:
        return None


def parse_cpulist(text):
    """Expand a sysfs/Slurm range list such as '32-63,96-127' into a set of ints."""
    cpus = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def format_ranges(values):
    """The inverse: {32,...,63} -> '32-63'."""
    out, values = [], sorted(values)
    while values:
        lo = hi = values.pop(0)
        while values and values[0] == hi + 1:
            hi = values.pop(0)
        out.append(str(lo) if lo == hi else f"{lo}-{hi}")
    return ",".join(out)


def core_map():
    """Logical CPU -> Slurm's abstract core index, or None if it cannot be built.

    gres.conf's Cores= is in core indices, not CPU indices.  On a hyperthreaded
    node those differ: sysfs local_cpulist reports logical CPUs, so a two-thread
    node yields values up to 2*cores-1 and Slurm rejects the ones past its core
    count.  Build the mapping from the kernel's own sibling sets rather than
    assuming how CPUs are enumerated, and number the cores by their lowest CPU,
    which is the order Slurm derives too.
    """
    siblings = {}
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/"
                          "thread_siblings_list"):
        try:
            with open(path) as f:
                group = frozenset(parse_cpulist(f.read().strip()))
        except OSError:
            return None
        cpu = int(re.search(r"/cpu(\d+)/", path).group(1))
        siblings[cpu] = group
    if not siblings:
        return None
    order = sorted({group for group in siblings.values()}, key=min)
    index = {group: i for i, group in enumerate(order)}
    return {cpu: index[group] for cpu, group in siblings.items()}


def socket_map():
    """(logical CPU -> socket, socket -> set of core indices), or None on failure."""
    mapping = core_map()
    if mapping is None:
        return None
    cpu_pkg, pkg_cores = {}, {}
    for cpu, core in mapping.items():
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/"
                      "physical_package_id") as f:
                pkg = int(f.read().strip())
        except (OSError, ValueError):
            return None
        cpu_pkg[cpu] = pkg
        pkg_cores.setdefault(pkg, set()).add(core)
    return (cpu_pkg, pkg_cores) if pkg_cores else None


def gpu_cores(pci):
    """The Cores= value for a GPU, or '' if it cannot be determined safely.

    Whole sockets, not the GPU's own NUMA node.  Slurm requires the core set to fall on
    socket boundaries and drains the node with INVALID_REG otherwise:

        gres/gpu GRES core specification 8-15 for node drp-srcf-gpu008 doesn't match
        socket boundaries. (Socket 0 is cores 0-31)

    A GPU's sysfs locality is finer than a socket wherever there is more than one NUMA
    node per socket.  drp-srcf-gpu008 is NPS=4, so local_cpulist gives eight cores of a
    thirty-two-core socket.  Nodes with one NUMA node per socket hide the distinction
    entirely -- on gpu006 (2x32) and gpu001 (2x8) the GPU's local cores were *exactly*
    one socket, so the narrow form and the socket form agreed and the narrow one looked
    correct.  Widening is therefore safe for those two as well as necessary here.

    The NUMA node stays in the comment above each record, so the finer locality is not
    lost -- it is just not something Slurm can express here.
    """
    cpulist = pci_attr(pci, "local_cpulist")
    if not cpulist:
        return ""
    maps = socket_map()
    if maps is None:
        return ""
    cpu_pkg, pkg_cores = maps
    try:
        cpus = parse_cpulist(cpulist)
    except ValueError:
        return ""
    pkgs = {cpu_pkg[cpu] for cpu in cpus if cpu in cpu_pkg}
    cores = set().union(*(pkg_cores[pkg] for pkg in pkgs)) if pkgs else set()
    # Omitting Cores= costs a scheduling hint; emitting a wrong one leaves the node's
    # gres configuration invalid, so say nothing rather than guess.
    return format_ranges(cores) if cores else ""


def pair(datadevs, gpus):
    """Pair each usable datadev with a GPU, preferring the closest one.

    Returns the pairings and the complaints, so that the caller decides how loud to
    be.  Deliberately does not silently drop a card: an unpaired datadev is capacity
    that would vanish without anyone noticing.
    """
    usable = [c for c in datadevs if c["gpu_capable"]]
    free   = [g for g in gpus if not g["excluded"]]
    paths  = {d["pci"]: pci_path(d["pci"]) for d in usable}
    paths.update({g["pci"]: pci_path(g["pci"]) for g in free})

    def best(card, pool):
        """The closest GPU in pool, and how close it is."""
        ranked = sorted(pool,
                        key=lambda g: shared_depth(paths[card["pci"]], paths[g["pci"]]),
                        reverse=True)
        if not ranked:
            return None, 0
        return ranked[0], shared_depth(paths[card["pci"]], paths[ranked[0]["pci"]])

    # Cards with a local option claim it first, so that a card with no local option
    # cannot consume the GPU another card needed
    pairs, notes = [], []
    remaining = list(usable)
    for want_local in (True, False):
        for card in list(remaining):
            gpu, depth = best(card, free)
            if not gpu:
                break
            if want_local and depth < LOCAL_DEPTH:
                continue                    # Leave it for the second pass
            free.remove(gpu)
            remaining.remove(card)
            pairs.append({"datadev": card, "gpu": gpu,
                          "local": depth >= LOCAL_DEPTH, "depth": depth})
            if depth < LOCAL_DEPTH:
                notes.append(f"{card['name']} ({card['pci']}) is paired with GPU "
                             f"{gpu['pci']}, which is not behind the same PCIe "
                             f"switch, so its DMA crosses a root complex")

    for card in datadevs:
        if not card["gpu_capable"]:
            notes.append(f"{card['name']} ({card['pci']}) has no GPU support "
                         f"(firmware '{card['firmware']}'); not offered as a gres")
    for card in remaining:
        notes.append(f"{card['name']} ({card['pci']}) has no GPU left to pair with; "
                     f"it CANNOT be used by drp_gpu on this node")
    for gpu in gpus:
        if gpu["excluded"]:
            notes.append(f"GPU {gpu['pci']} is excluded by the driver; ignored")
        elif gpu in free:
            notes.append(f"GPU {gpu['pci']} ({gpu['dev']}) is spare -- no datadev "
                         f"needs it")
    return pairs, notes


# Blank lines are part of the markers: pasted between other nodes' records, a block
# that touches them is hard to see the extent of, and the extent is what you need when
# replacing one.
BEGIN = "\n# >>> gen_gres_conf.py: {node} >>>"
END   = "# <<< gen_gres_conf.py: {node} <<<\n"


def emit(pairs, node):
    """This node's records, and only what a reader of gres.conf needs in place.

    Deliberately terse.  The explanation of why the pairing exists, why the datadev is
    not a gres and why AutoDetect is off lives on the Confluence page, because this
    block is repeated per node: twenty nodes' worth of identical preamble would make
    the file unreadable, which is the thing that argues against generating it at all.
    What stays is what differs between nodes -- the pairings -- plus provenance.
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # One header line, because this repeats per node.  It records the command rather
    # than describing the result: the natural thing to do with a file like this is to
    # copy a neighbouring line and edit it, and File= and Cores= cannot be guessed that
    # way, so what a reader most needs is the command that would produce them.  It also
    # carries the --exclude, without which a rerun silently pairs a card that should
    # have been left out.  The reasoning is on the Confluence page, not repeated here.
    cmd = " ".join(shlex.quote(arg)
                   for arg in [os.path.basename(sys.argv[0])] + sys.argv[1:])
    out = [BEGIN.format(node=node),
           f"# Generated {stamp} by {getpass.getuser()} on {node}: '{cmd}'."
           f"  Rerun there; do not hand-edit.  Confluence: GPU DRP > gres.conf."]

    gres_counts = {}
    for p in sorted(pairs, key=lambda p: p["datadev"]["pci"]):
        card, gpu = p["datadev"], p["gpu"]
        typ   = f"dd{card['pci'].split(':')[1]}"
        cores = gpu_cores(gpu["pci"])
        numa  = pci_attr(gpu["pci"], "numa_node") or "?"
        where = "same PCIe switch" if p["local"] else "*** DIFFERENT PCIe switch ***"
        suffix = f" Cores={cores}" if cores else ""
        out += [f"# {card['name']} {card['pci']} ({card['dev']}) <-> GPU {gpu['pci']} "
                f"({gpu['dev']}), {where}, NUMA {numa}",
                f"NodeName={node} AutoDetect=off Name=gpu Type={typ} "
                f"File={gpu['dev']}{suffix}"]
        gres_counts[f"gpu:{typ}"] = 1

    out.append(END.format(node=node))
    return "\n".join(out) + "\n", ",".join(f"{k}:1" for k in sorted(gres_counts))


CACHED_CONF = "/var/spool/slurmd/conf-cache/gres.conf"

# --check's verdict is the one line a person is actually looking for, in among the
# warnings and counts, so it gets a blank line and a colour.  Only when stderr is a
# terminal: the escape codes must not end up in whatever captured the output, which is
# the case that matters when it fails.
GREEN = "1;32"
RED   = "1;31"


def _verdict(text, colour):
    if not sys.stderr.isatty():
        return text
    return f"\033[{colour}m{text}\033[0m"


def _records(text, node):
    """The substantive NodeName= lines for a node, ignoring comments and spacing."""
    mine = re.compile(r"^\s*NodeName=" + re.escape(node) + r"(?=\s)", re.I)
    return [" ".join(l.split()) for l in text.splitlines()
            if not l.strip().startswith("#") and mine.match(l)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", metavar="FILE",
                    help="write here instead of stdout")
    ap.add_argument("-e", "--expect", type=int, metavar="N",
                    help="the number of datadev/GPU pairs this node is supposed to "
                         "have.  Print nothing if the count differs, in either "
                         "direction: a node quietly coming up short is how science "
                         "data acquires a hole in the image, and a node with more "
                         "pairs than anyone expected is not understood either.  Omit "
                         "it to see what is actually there")
    ap.add_argument("-x", "--exclude", metavar="CARDS",
                    help="comma-separated datadev cards to leave out of the pairing, "
                         "by bus number ('84'), device name ('datadev_84') or PCI "
                         "address.  For a card that is present and GPU-capable but "
                         "has no GPU DRP role, so that it does not consume a GPU "
                         "another card needs.  A name that matches nothing is an "
                         "error, not a no-op")
    ap.add_argument("-c", "--check", action="store_true",
                    help="compare this node's hardware against what the gres.conf it "
                         "was given already says, and exit non-zero if they differ.  "
                         "Writes nothing.  Reads only /proc, /sys and the local "
                         + CACHED_CONF)
    args = ap.parse_args()

    datadevs = read_datadevs()
    gpus     = read_gpus()
    if not datadevs:
        print("error: no /proc/datadev_* found; is the datadev driver loaded?",
              file=sys.stderr)
        return 1
    if not gpus:
        print(f"error: no GPUs found under {PROC_NVIDIA}; is the NVIDIA driver "
              f"loaded?", file=sys.stderr)
        return 1

    excluded = []
    if args.exclude:
        wanted = {tok.strip().lower() for tok in args.exclude.split(",") if tok.strip()}
        def aliases(card):
            name = card["name"].lower()            # datadev_84
            return {name, name[len("datadev_"):], card["pci"]}
        matched = {tok for card in datadevs for tok in wanted & aliases(card)}
        # A typo must not pass quietly: the card would stay in the pairing, and with
        # --expect satisfied by the wrong set of cards the result looks correct.
        if wanted - matched:
            print(f"error: --exclude names no such datadev card: "
                  f"{', '.join(sorted(wanted - matched))}.  Present: "
                  f"{', '.join(card['name'] for card in datadevs)}", file=sys.stderr)
            return 2
        excluded = [card for card in datadevs if wanted & aliases(card)]
        datadevs = [card for card in datadevs if not wanted & aliases(card)]
        if not datadevs:
            print("error: --exclude leaves no datadev card at all", file=sys.stderr)
            return 2

    # Before pairing, or the driver's single fault is reported once per card as a
    # firmware complaint, sending the reader off to reflash FPGAs that are fine.
    if not any(card["gpu_driver"] for card in datadevs):
        print(f"error: the loaded datadev driver was built without DATA_GPU "
              f"('GPUAsync Support : Disabled' in /proc/datadev_*), so no card can "
              f"report GPU capability whatever its firmware.  Install the driver "
              f"built with NVIDIA_DRIVERS set.  Both builds are datadev.ko, so "
              f"lsmod cannot tell them apart.", file=sys.stderr)
        return 1

    pairs, notes = pair(datadevs, gpus)
    remote = sum(1 for p in pairs if not p["local"])
    counts = (f"{len(datadevs)} datadev card(s), {len(gpus)} GPU(s), "
              f"{len(pairs)} pair(s), {remote} not on a shared PCIe switch")
    if excluded:
        counts += (f", excluded by request: "
                   f"{' '.join(card['name'] for card in excluded)}")

    for note in notes:
        print(f"warning: {note}", file=sys.stderr)
    print(f"info: {counts}", file=sys.stderr)

    node = os.uname().nodename

    if args.check:
        try:
            with open(CACHED_CONF) as f:
                have = _records(f.read(), node)
        except OSError as exc:
            print(f"error: cannot read {CACHED_CONF}: {exc}", file=sys.stderr)
            return 1
        want = _records(emit(pairs, node)[0], node)
        print(file=sys.stderr)
        if have == want:
            print(_verdict(f"info: {CACHED_CONF} matches this node's hardware "
                           f"({len(want)} record(s))", GREEN), file=sys.stderr)
            return 0
        print(_verdict(f"error: {CACHED_CONF} does not match this node's hardware",
                       RED), file=sys.stderr)
        for line in sorted(set(have) - set(want)):
            print(f"error:   configured but not present: {line}", file=sys.stderr)
        for line in sorted(set(want) - set(have)):
            print(f"error:   present but not configured: {line}", file=sys.stderr)
        return 1

    # No pairs is never a configuration worth publishing, and --expect is optional,
    # so this is checked independently of it.
    if not pairs:
        print(f"error: no datadev card could be paired with a GPU, so this node "
              f"cannot run drp_gpu at all.  The warnings above say which cards were "
              f"rejected and why.", file=sys.stderr)
        return 1

    if args.expect is not None and len(pairs) != args.expect:
        print(f"error: expected {args.expect} pairs but found {len(pairs)}.  "
              f"Printing nothing rather than a configuration for a node whose "
              f"hardware is not what it is supposed to be.  Rerun without --expect "
              f"to see it anyway.", file=sys.stderr)
        return 1

    text, gres_spec = emit(pairs, node)
    if args.output:
        tmp = args.output + ".new"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, args.output)
        print(f"info: wrote {args.output}", file=sys.stderr)
        source = args.output
    else:
        sys.stdout.write(text)
        source = "the block above"
    # The instructions go on stderr, not into the block: they are for whoever is
    # publishing today, whereas gres.conf is read by someone else months later.
    print(f"info: paste {source} into psslurmctld001:/etc/slurm/gres.conf, replacing "
          f"any existing {node} lines", file=sys.stderr)
    print(f"info: and set this node's line in slurm.conf to match:", file=sys.stderr)
    print(f"info:   NodeName={node} ... Gres={gres_spec}", file=sys.stderr)
    print(f"info: then 'sudo scontrol reconfigure' there.  Converting a node drains it; "
          f"clear that with", file=sys.stderr)
    print(f"info:   sudo scontrol update NodeName={node} State=RESUME",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
