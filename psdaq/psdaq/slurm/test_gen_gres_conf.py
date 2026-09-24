"""Tests for gen_gres_conf, against synthetic sysfs and /proc trees.

Why a fake tree rather than the real one.  Two behaviours here cannot be exercised on
any machine we have:

  * `Cores=` must name whole *sockets*, not the GPU's NUMA node.  The two differ only
    when a socket holds more than one NUMA node, i.e. NPS>1 on the EPYC boxes.  Every
    node was moved to NPS=1 on 2026-09-22/23, so a NUMA node now *is* a socket
    everywhere and the wrong definition and the right one agree.  drp-srcf-gpu008 at
    NPS=4 used to be the only node that could catch a regression; these tests are now
    the only guard.

  * A GPU that is listed but unusable -- `Bus Type: PCI` and an unreadable
    `current_link_speed`, which is what a failed GPU whose removal was refused looks
    like.  Reproducing that needs a broken GPU on demand.

So the tests build `/sys` and `/proc` fragments in a temporary directory and rebind
the module's path constants to them.  Everything asserted is the module's own output;
nothing is mocked at the function level, so the tests fail if the walk over sysfs
changes shape.

Run with `pytest test_gen_gres_conf.py`, or directly for a plain summary.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from psdaq.slurm import gen_gres_conf as G


# ---------------------------------------------------------------- fake trees

def write(path, text):
    """Create `path`'s parents and write `text` to it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def fake_cpus(tmp, sockets, cores_per_socket, threads):
    """A /sys/devices/system/cpu tree, Linux's own enumeration.

    Machine CPU n and n + total/threads are siblings, and the first half is laid out
    socket-major -- the arrangement every node here reports.
    """
    root = os.path.join(tmp, "cpu")
    total = sockets * cores_per_socket * threads
    half = total // threads
    for cpu in range(total):
        lin = cpu % half                      # position among first siblings
        pkg = lin // cores_per_socket
        sibs = ",".join(str(lin + i * half) for i in range(threads))
        topo = os.path.join(root, f"cpu{cpu}", "topology")
        write(os.path.join(topo, "thread_siblings_list"), sibs + "\n")
        write(os.path.join(topo, "physical_package_id"), f"{pkg}\n")
    return root


def fake_pci(tmp, devices):
    """A /sys/bus/pci/devices tree.

    `devices` maps a PCI address to a dict of attribute -> contents.  An attribute
    whose value is None is left absent, which is how an unreadable
    `current_link_speed` presents.
    """
    root = os.path.join(tmp, "pci")
    for pci, attrs in devices.items():
        for attr, text in attrs.items():
            if text is None:
                continue
            write(os.path.join(root, pci, attr), text + "\n")
        os.makedirs(os.path.join(root, pci), exist_ok=True)
    return root


def fake_nvidia(tmp, gpus):
    """A /proc/driver/nvidia/gpus tree.  `gpus` maps PCI address to `information`."""
    root = os.path.join(tmp, "nvidia")
    for pci, text in gpus.items():
        write(os.path.join(root, pci, "information"), text)
    return root


def information(minor, bus_type="PCIe", excluded="No"):
    """The fields of /proc/driver/nvidia/gpus/*/information that we read."""
    lines = ["Model: \t\t NVIDIA H200 NVL",
             f"Device Minor: \t {minor}",
             f"GPU Excluded:\t {excluded}"]
    if bus_type is not None:
        lines.insert(1, f"Bus Type: \t {bus_type}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    """Rebind the module's path constants; returns a builder for each tree."""
    class Builder:
        def cpus(self, sockets, cores_per_socket, threads=1):
            monkeypatch.setattr(G, "SYS_CPU",
                                fake_cpus(str(tmp_path), sockets, cores_per_socket, threads))

        def pci(self, devices):
            monkeypatch.setattr(G, "SYS_PCI", fake_pci(str(tmp_path), devices))

        def nvidia(self, gpus):
            monkeypatch.setattr(G, "PROC_NVIDIA", fake_nvidia(str(tmp_path), gpus))

    return Builder()


# ---------------------------------------------------- core and socket mapping

@pytest.mark.parametrize("sockets,cores,threads", [(2, 32, 1), (2, 8, 1), (2, 16, 2), (2, 32, 2)])
def test_core_map_numbers_every_core_once(sysfs, sockets, cores, threads):
    """Each core gets one index in 0..cores-1, and siblings share it."""
    sysfs.cpus(sockets, cores, threads)
    mapping = G.core_map()
    assert mapping is not None
    assert len(mapping) == sockets * cores * threads       # every CPU present
    assert set(mapping.values()) == set(range(sockets * cores))
    if threads == 2:
        half = sockets * cores
        for cpu in range(half):
            assert mapping[cpu] == mapping[cpu + half], "siblings must share a core index"


def test_socket_map_splits_cores_evenly(sysfs):
    sysfs.cpus(2, 32, 1)
    cpu_pkg, pkg_cores = G.socket_map()
    assert sorted(pkg_cores) == [0, 1]
    assert pkg_cores[0] == set(range(0, 32))
    assert pkg_cores[1] == set(range(32, 64))
    assert cpu_pkg[0] == 0 and cpu_pkg[63] == 1


def test_core_map_is_none_without_sysfs(sysfs, tmp_path):
    """No topology at all is a failure, not an empty answer."""
    sysfs.cpus(2, 8, 1)
    G.SYS_CPU = str(tmp_path / "absent")
    assert G.core_map() is None
    assert G.socket_map() is None


# ------------------------------------------------- Cores= on socket boundaries

def test_gpu_cores_widens_to_a_whole_socket_under_nps4(sysfs):
    """The regression no node can catch any more.

    NPS=4: the GPU's local_cpulist is eight cores of a thirty-two-core socket.  Slurm
    requires the set to fall on socket boundaries and drains the node otherwise, so
    the narrow NUMA-node answer is wrong and the whole socket is right.
    """
    sysfs.cpus(2, 32, 1)
    sysfs.pci({"0000:d4:00.0": {"local_cpulist": "40-47"}})     # one NPS=4 quadrant
    assert G.gpu_cores("0000:d4:00.0") == "32-63"               # not "40-47"


def test_gpu_cores_second_socket_under_nps1(sysfs):
    """With one NUMA node per socket the narrow and wide answers coincide."""
    sysfs.cpus(2, 32, 1)
    sysfs.pci({"0000:d4:00.0": {"local_cpulist": "32-63"}})
    assert G.gpu_cores("0000:d4:00.0") == "32-63"


def test_gpu_cores_is_core_indices_not_cpu_indices(sysfs):
    """With SMT on, local_cpulist runs to 2*cores-1; Cores= must not."""
    sysfs.cpus(2, 32, 2)                                        # 128 CPUs, 64 cores
    sysfs.pci({"0000:d4:00.0": {"local_cpulist": "32-63,96-127"}})
    assert G.gpu_cores("0000:d4:00.0") == "32-63"               # cores, not CPUs


def test_gpu_cores_spanning_two_sockets_gives_both(sysfs):
    sysfs.cpus(2, 8, 1)
    sysfs.pci({"0000:81:00.0": {"local_cpulist": "4-12"}})      # straddles the boundary
    assert G.gpu_cores("0000:81:00.0") == "0-15"


@pytest.mark.parametrize("cpulist", [None, "", "not-a-list"])
def test_gpu_cores_says_nothing_rather_than_guessing(sysfs, cpulist):
    """A wrong Cores= invalidates the node's gres; omitting it costs only a hint."""
    sysfs.cpus(2, 8, 1)
    sysfs.pci({"0000:81:00.0": {"local_cpulist": cpulist}})
    assert G.gpu_cores("0000:81:00.0") == ""


# ------------------------------------------------------- the dead-GPU guard

HEALTHY = {"current_link_speed": "16.0 GT/s PCIe"}
DEAD    = {"current_link_speed": None}          # EINVAL once the PCIe caps are gone


def test_read_gpus_accepts_a_healthy_gpu(sysfs):
    sysfs.nvidia({"0000:03:00.0": information(minor=0)})
    sysfs.pci({"0000:03:00.0": HEALTHY})
    gpus = G.read_gpus()
    assert [g["minor"] for g in gpus] == [0]
    assert gpus[0]["dev"] == "/dev/nvidia0"
    assert gpus[0]["excluded"] is False


def test_read_gpus_skips_one_that_is_listed_but_dead(sysfs, capsys):
    """Both symptoms together: 'Bus Type: PCI' and no readable link speed.

    Observed on drp-srcf-gpu008 after an Xid 154 GSP hang on 2026-09-17: the driver
    asked the kernel to remove the GPU, something held it open, the removal was
    refused, and /proc went on listing it.  Pairing a card with it yields a gres
    record Slurm allocates happily and a DRP that dies in CUDA init.
    """
    sysfs.nvidia({"0000:03:00.0": information(minor=0),
                  "0000:d4:00.0": information(minor=1, bus_type="PCI")})
    sysfs.pci({"0000:03:00.0": HEALTHY, "0000:d4:00.0": DEAD})
    gpus = G.read_gpus()
    assert [g["pci"] for g in gpus] == ["0000:03:00.0"], "the dead GPU must not be offered"
    assert "not usable" in capsys.readouterr().err


def test_read_gpus_keeps_a_gpu_with_only_one_symptom(sysfs):
    """Requiring both symptoms avoids discarding a GPU over one odd reading."""
    # Bus Type wrong, link speed readable.
    sysfs.nvidia({"0000:d4:00.0": information(minor=0, bus_type="PCI")})
    sysfs.pci({"0000:d4:00.0": HEALTHY})
    assert len(G.read_gpus()) == 1

    # Bus Type right, link speed unreadable.
    sysfs.nvidia({"0000:d4:00.0": information(minor=0)})
    sysfs.pci({"0000:d4:00.0": DEAD})
    assert len(G.read_gpus()) == 1


def test_read_gpus_treats_a_missing_bus_type_as_healthy(sysfs):
    """Absence means the field moved or was renamed, not that the GPU failed."""
    sysfs.nvidia({"0000:d4:00.0": information(minor=0, bus_type=None)})
    sysfs.pci({"0000:d4:00.0": DEAD})
    assert len(G.read_gpus()) == 1


def test_read_gpus_notes_an_excluded_gpu_without_dropping_it(sysfs):
    sysfs.nvidia({"0000:03:00.0": information(minor=0, excluded="Yes")})
    sysfs.pci({"0000:03:00.0": HEALTHY})
    gpus = G.read_gpus()
    assert len(gpus) == 1 and gpus[0]["excluded"] is True


def test_read_gpus_skips_an_entry_with_no_device_minor(sysfs, capsys):
    sysfs.nvidia({"0000:03:00.0": "Model: \t NVIDIA H200 NVL\nBus Type: \t PCIe\n"})
    sysfs.pci({"0000:03:00.0": HEALTHY})
    assert G.read_gpus() == []
    assert "no Device Minor" in capsys.readouterr().err


# ------------------------------------------------------------------ helpers

@pytest.mark.parametrize("text,expected", [
    ("32-63", set(range(32, 64))),
    ("0,4-6", {0, 4, 5, 6}),
    ("32-63,96-127", set(range(32, 64)) | set(range(96, 128))),
    ("7", {7}),
])
def test_parse_cpulist(text, expected):
    assert G.parse_cpulist(text) == expected


@pytest.mark.parametrize("values,expected", [
    (range(0, 16), "0-15"),
    ([0, 1, 2, 8, 9], "0-2,8-9"),
    ([5], "5"),
    ([], ""),
])
def test_format_ranges(values, expected):
    assert G.format_ranges(values) == expected


def test_format_ranges_round_trips_parse_cpulist():
    for text in ("0-31", "0-2,8-9", "5", "32-63,96-127"):
        assert G.format_ranges(G.parse_cpulist(text)) == text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
