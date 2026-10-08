"""Device selection runs before MPI and refuses to guess.

Phase 1 is deliberately host-only, so these tests need no GPU: the identity
table and the launcher environment are both injected.
"""
import os
from types import SimpleNamespace as NS

import pytest

from psana.gpu import gpu_placement as gp
from psana.gpu.gpu_placement import (
    GpuPlacement, GpuPlacementError, PinnedDevice, LOCAL_RANK_VARS,
    per_rank_limit, pin_device, uuid_hex, verify_pin,
)


TWO_GPUS = {
    0: {'uuid': 'GPU-aaaaaaaa-1111-2222-3333-444444444444',
        'pci': '0000:01:00.0', 'mig': False},
    1: {'uuid': 'GPU-bbbbbbbb-5555-6666-7777-888888888888',
        'pci': '0000:41:00.0', 'mig': False},
}

MIG_PAIR = {
    0: {'uuid': 'MIG-cccccccc-1111-2222-3333-444444444444',
        'pci': '0000:07:00.0', 'mig': True},
    1: {'uuid': 'MIG-dddddddd-5555-6666-7777-888888888888',
        'pci': '0000:07:00.0', 'mig': True},   # same bus id, separate memory
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in LOCAL_RANK_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.delenv('CUDA_DEVICE_ORDER', raising=False)
    # The test process itself imports mpi4py (psana's conftest), so the
    # strict-mode guard would fire in every test. Hide the CUDA initialisers
    # by default; the two tests that exercise the guard re-insert them.
    monkeypatch.setattr(gp, 'CUDA_INITIALISERS', ())


def use_devices(monkeypatch, table, source='nvidia-smi'):
    monkeypatch.setattr(gp, 'node_devices', lambda: (dict(table), source))


# ---------------------------------------------------------------------------
# The ordering constraint
# ---------------------------------------------------------------------------

def test_pin_does_not_narrow_the_mask(monkeypatch):
    """Choosing, not masking.

    `import psana` loads mpi4py and initialises CUDA, so a
    CUDA_VISIBLE_DEVICES write would be ignored in any real job. The device is
    selected later with Device.use() instead, which was measured sufficient:
    with both GPUs visible, every rank's allocations stayed on its own device.
    """
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setenv('SLURM_LOCALID', '1')
    pinned = pin_device()
    assert pinned.pinned
    assert pinned.requested_pci == TWO_GPUS[1]['pci']
    # The launcher's mask is left exactly as it was.
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'


def test_pin_works_with_cuda_already_initialised(monkeypatch):
    """No guard, no warning: arriving after CUDA init is the normal case."""
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setitem(gp.sys.modules, 'cupy', NS())
    monkeypatch.setitem(gp.sys.modules, 'mpi4py.MPI', NS())
    monkeypatch.setenv('SLURM_LOCALID', '0')
    pinned = pin_device()
    assert pinned.pinned and not pinned.warnings


def test_pin_sets_deterministic_device_order(monkeypatch):
    """NVML numbers by PCI bus, CUDA defaults to FASTEST_FIRST. Identical
    cards usually agree, which is luck rather than a guarantee."""
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('SLURM_LOCALID', '0')
    pin_device()
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'


# ---------------------------------------------------------------------------
# Choosing within the launcher's allocation
# ---------------------------------------------------------------------------

def test_node_local_rank_selects_distinct_devices(monkeypatch):
    use_devices(monkeypatch, TWO_GPUS)
    chosen = []
    for rank in range(4):
        # Reset the mask each time: in production every rank is a separate
        # process, and pin_device() narrows the mask to the chosen UUID.
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
        monkeypatch.setenv('SLURM_LOCALID', str(rank))
        chosen.append(pin_device().requested_uuid)
    # Round-robin over the permitted set, so peers land two per device.
    assert chosen[0] == chosen[2] == TWO_GPUS[0]['uuid']
    assert chosen[1] == chosen[3] == TWO_GPUS[1]['uuid']


def test_pin_is_idempotent_and_side_effect_free(monkeypatch):
    """Repeated calls return the same answer for the same inputs.

    Now that the mask is left alone, pin_device() is a pure function of the
    environment and the device table -- so it can be called twice, or called
    again after a transition, without changing what it reports.
    """
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setenv('SLURM_LOCALID', '1')
    first = pin_device()
    second = pin_device()
    assert second.requested_pci == first.requested_pci
    assert second.permitted == first.permitted == ('0', '1')


def test_identity_is_recorded_by_pci_and_uuid(monkeypatch):
    """Both are recorded: the PCI id is what select_device and verify_pin
    match on, and the UUID is the grouping key -- read from NVML here rather
    than from CuPy, which truncates it at the first NUL byte."""
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setenv('SLURM_LOCALID', '1')
    pinned = pin_device()
    assert pinned.requested_pci == TWO_GPUS[1]['pci']
    assert pinned.requested_uuid == TWO_GPUS[1]['uuid']


def test_single_entry_mask_is_honoured(monkeypatch):
    """--gpus-per-task=1 already assigns one device; overwriting it is wrong."""
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    monkeypatch.setenv('SLURM_LOCALID', '3')
    pinned = pin_device()
    assert pinned.requested_uuid == TWO_GPUS[1]['uuid']
    assert pinned.permitted == ('1',)


def test_unset_mask_considers_every_node_device(monkeypatch):
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('SLURM_LOCALID', '1')
    pinned = pin_device()
    assert pinned.permitted == ('0', '1')
    assert pinned.requested_uuid == TWO_GPUS[1]['uuid']


def test_uuid_entries_in_the_mask_resolve(monkeypatch):
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', TWO_GPUS[1]['uuid'])
    monkeypatch.setenv('SLURM_LOCALID', '0')
    assert pin_device().requested_uuid == TWO_GPUS[1]['uuid']


@pytest.mark.parametrize('var', LOCAL_RANK_VARS)
def test_every_launcher_variable_is_accepted(monkeypatch, var):
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setenv(var, '1')
    pinned = pin_device()
    assert pinned.local_rank_source == var
    assert pinned.requested_uuid == TWO_GPUS[1]['uuid']


def test_launcher_variable_preference_order(monkeypatch):
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('SLURM_LOCALID', '1')
    monkeypatch.setenv('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    assert pin_device().local_rank_source == 'OMPI_COMM_WORLD_LOCAL_RANK'


def test_missing_local_rank_warns_because_all_ranks_would_collide(monkeypatch):
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    pinned = pin_device()
    assert pinned.local_rank == 0
    assert any('local-rank' in w for w in pinned.warnings)


# ---------------------------------------------------------------------------
# Degradation, never a wrong device
# ---------------------------------------------------------------------------

def test_absent_index_is_reported_not_guessed(monkeypatch):
    """Slurm ConstrainDevices=yes can renumber the mask relative to the
    allocation while NVML reports physical indices."""
    use_devices(monkeypatch, {0: TWO_GPUS[0]})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    monkeypatch.setenv('SLURM_LOCALID', '0')
    pinned = pin_device()
    assert not pinned.pinned
    assert any('cgroup renumbering' in w for w in pinned.warnings)


def test_no_identity_available_does_not_pin(monkeypatch):
    monkeypatch.setattr(gp, 'node_devices', lambda: ({}, ''))
    pinned = pin_device()
    assert not pinned.pinned and not pinned.usable
    assert 'CUDA_VISIBLE_DEVICES' not in os.environ


def test_empty_mask_does_not_pin(monkeypatch):
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('SLURM_LOCALID', '0')
    # An empty string means "no devices", not "all devices". This assertion
    # previously read ('0', '1') -- the comment was right and the assertion
    # encoded the bug, which is how it survived review.
    assert pin_device().permitted == ()


def test_mig_is_detected_from_the_identity_source(monkeypatch):
    """Raw UUID bytes carry no GPU/MIG discriminator, so MIG-ness can only
    come from NVML or nvidia-smi."""
    use_devices(monkeypatch, MIG_PAIR)
    monkeypatch.setenv('SLURM_LOCALID', '0')
    assert pin_device().is_mig is True


# ---------------------------------------------------------------------------
# Verification after the CuPy import
# ---------------------------------------------------------------------------

def fake_cp(pci, *, visible=1, ordinal=0):
    """Stand-in exposing only what verify_pin uses.

    Note it returns a PCI bus id, not a UUID: verify_pin identifies the device
    through deviceGetPCIBusId because getDeviceProperties()['uuid'] is a
    16-byte array that CuPy truncates at the first NUL byte -- measured on
    sdfampere014, where a UUID containing 0x00 produced a 5-byte value.
    """
    return NS(cuda=NS(
        runtime=NS(getDeviceCount=lambda: visible,
                   deviceGetPCIBusId=lambda i: pci),
        Device=lambda *a: NS(id=ordinal, use=lambda: None)))


# Normalised form: _pci_norm() strips the domain's leading zeros so that
# nvidia-smi's eight digits and CUDA's four compare equal.
PCI_A = '0:01:00.0'
PCI_B = '0:41:00.0'


def pinned_at(pci, uuid='GPU-' + 'a' * 32):
    return PinnedDevice(requested_uuid=uuid, requested_pci=pci, pinned=True)


def test_verify_accepts_a_pin_that_took_effect():
    pinned = pinned_at(PCI_A)
    uuid, ordinal, visible, errors = verify_pin(pinned, fake_cp(PCI_A))
    assert errors == [] and visible == 1
    # The full recorded UUID is returned, never a truncated driver value.
    assert uuid == pinned.requested_uuid


def test_verify_tolerates_a_wide_visible_set():
    """The mask is never narrowed, and a wide visible set was measured not to
    cause misplaced allocations, so the count is logged rather than checked."""
    _, _, visible, errors = verify_pin(pinned_at(PCI_A),
                                       fake_cp(PCI_A, visible=4))
    assert errors == [] and visible == 4


def test_verify_rejects_a_device_this_rank_did_not_select():
    """The one rule: the current device's PCI bus id must be the chosen one."""
    _, _, _, errors = verify_pin(pinned_at(PCI_A), fake_cp(PCI_B))
    assert any('did not select' in e for e in errors)


def test_verify_rejects_the_wrong_device_however_many_are_visible():
    _, _, _, errors = verify_pin(pinned_at(PCI_A), fake_cp(PCI_B, visible=4))
    assert any('did not select' in e for e in errors)


def test_verify_is_silent_when_no_pin_was_applied():
    uuid, _, _, errors = verify_pin(PinnedDevice(), fake_cp(PCI_A, visible=4))
    assert errors == []
    # With nothing pinned, the PCI id is the only identity available.
    assert uuid == PCI_A


def test_uuid_survives_a_nul_byte(monkeypatch):
    """A UUID containing 0x00 must not be truncated.

    CuPy's getDeviceProperties()['uuid'] loses everything after the first NUL;
    sdfampere014 has such a device and the old code hashed it to 8 characters.
    verify_pin now carries the recorded UUID through instead.
    """
    uuid = 'GPU-c9ef9054-6100-fd2e-2ec9-b5b06e54af77'
    assert '00' in uuid_hex(uuid)                    # contains a zero byte
    got, _, _, errors = verify_pin(pinned_at(PCI_A, uuid), fake_cp(PCI_A))
    assert errors == [] and got == uuid


def test_select_device_finds_the_requested_pci():
    from psana.gpu.gpu_placement import select_device
    used = []
    cp = NS(cuda=NS(
        runtime=NS(getDeviceCount=lambda: 2,
                   deviceGetPCIBusId=lambda i: (PCI_A, PCI_B)[i]),
        Device=lambda i: NS(id=i, use=lambda: used.append(i))))
    assert select_device(pinned_at(PCI_B), cp) is True
    assert used == [1]


def test_select_device_reports_an_absent_pci():
    from psana.gpu.gpu_placement import select_device
    cp = NS(cuda=NS(
        runtime=NS(getDeviceCount=lambda: 1,
                   deviceGetPCIBusId=lambda i: PCI_A),
        Device=lambda i: NS(id=i, use=lambda: None)))
    assert select_device(pinned_at(PCI_B), cp) is False


def test_uuid_comparison_ignores_prefixes():
    assert uuid_hex('GPU-aaaa-bbbb') == uuid_hex('MIG-AAAA-BBBB') == 'aaaabbbb'


# ---------------------------------------------------------------------------
# Budget arithmetic
# ---------------------------------------------------------------------------

def test_limit_divides_the_device_among_true_peers():
    placement = GpuPlacement(pinned=PinnedDevice(), usable_bytes=4000,
                             n_device_peers=4)
    assert per_rank_limit(placement) == 1000


def test_shared_bytes_are_subtracted_once_not_per_rank():
    """Private bytes are already charged to their holder; subtracting them
    again would double-count."""
    placement = GpuPlacement(pinned=PinnedDevice(), usable_bytes=4000,
                             n_device_peers=4, shared_bytes=400)
    assert per_rank_limit(placement) == 900


def test_limit_never_goes_negative():
    placement = GpuPlacement(pinned=PinnedDevice(), usable_bytes=100,
                             n_device_peers=2, shared_bytes=500)
    assert per_rank_limit(placement) == 0


def test_can_share_requires_real_peers_on_a_full_gpu():
    base = dict(pinned=PinnedDevice(), device_comm=NS(), n_device_peers=2)
    assert GpuPlacement(**base).can_share
    assert not GpuPlacement(**{**base, 'n_device_peers': 1}).can_share
    assert not GpuPlacement(**{**base, 'device_comm': None}).can_share
    mig = GpuPlacement(**{**base, 'pinned': PinnedDevice(is_mig=True)})
    assert not mig.can_share          # IPC does not span MIG instances


def test_describe_reports_what_cannot_be_inferred_afterwards():
    placement = GpuPlacement(
        pinned=PinnedDevice(local_rank=2, local_rank_source='SLURM_LOCALID',
                            permitted=('0', '1')),
        hostname='sdfampere001', device_uuid_hex='abcdef0123456789' * 2,
        device_ordinal=0, visible_count=1, n_device_peers=2, is_owner=True,
        usable_bytes=18 * 1024 ** 3)
    line = placement.describe()
    for token in ('sdfampere001', 'peers=2', 'owner=True', 'visible=1',
                  'SLURM_LOCALID', '18.00GiB'):
        assert token in line


# ---------------------------------------------------------------------------
# MIG: a PCI bus id no longer identifies a device
# ---------------------------------------------------------------------------

def test_select_device_refuses_an_ambiguous_pci_id():
    """MIG instances of one card share a bus id.

    Taking the first match would select the wrong instance and hand this rank
    another rank's memory. CUDA's own UUID cannot break the tie because
    getDeviceProperties() truncates it at the first NUL byte, so the only safe
    answer is to refuse.
    """
    from psana.gpu.gpu_placement import select_device
    used = []
    cp = NS(cuda=NS(
        runtime=NS(getDeviceCount=lambda: 2,
                   # Two MIG instances on one physical card.
                   deviceGetPCIBusId=lambda i: PCI_A),
        Device=lambda i: NS(id=i, use=lambda: used.append(i))))
    with pytest.raises(GpuPlacementError, match='share pci'):
        select_device(pinned_at(PCI_A), cp)
    assert used == [], 'a device was selected despite the ambiguity'


def test_select_device_message_names_the_remedy():
    """The error has to say what to do: narrow the mask before Python starts."""
    from psana.gpu.gpu_placement import select_device
    cp = NS(cuda=NS(
        runtime=NS(getDeviceCount=lambda: 3,
                   deviceGetPCIBusId=lambda i: PCI_A),
        Device=lambda i: NS(id=i, use=lambda: None)))
    with pytest.raises(GpuPlacementError) as caught:
        select_device(pinned_at(PCI_A), cp)
    message = str(caught.value)
    assert 'MIG' in message
    assert 'CUDA_VISIBLE_DEVICES' in message


def test_select_device_is_unambiguous_when_the_mask_is_narrowed():
    """One visible device is the documented remedy, and it must work."""
    from psana.gpu.gpu_placement import select_device
    used = []
    cp = NS(cuda=NS(
        runtime=NS(getDeviceCount=lambda: 1,
                   deviceGetPCIBusId=lambda i: PCI_A),
        Device=lambda i: NS(id=i, use=lambda: used.append(i))))
    assert select_device(pinned_at(PCI_A), cp) is True
    assert used == [0]


def test_pin_device_docstring_states_one_identification_scheme():
    """The docstring previously claimed both PCI-id and UUID pinning."""
    doc = pin_device.__doc__
    assert 'PCI bus id' in doc
    # The mask write was removed, so nothing "pins by UUID" any more.
    assert 'Pinning by UUID' not in doc


def test_mig_uuid_mask_warns_that_it_could_not_pin(monkeypatch):
    """A MIG instance UUID in the mask is not in the node device map.

    NVML and `nvidia-smi --query-gpu` enumerate parent GPUs only, so the
    lookup misses, the rank is unpinned, and grouping falls back to the PCI
    bus id -- which MIG instances of one card share. Ranks on *different*
    instances then look like peers: sharing is attempted, IPC cannot span
    instances, and the budget is divided by too many peers.

    Safe but wasteful, and the index branch already warned on a miss while
    this branch said nothing. Checked here because MIG is untested hardware:
    the warning is the only thing that would explain the wasted memory.
    """
    parent_only = {0: {'uuid': 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
                       'pci': '0000:07:00.0', 'mig': True}}
    use_devices(monkeypatch, parent_only)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',
                       'MIG-11111111-2222-3333-4444-555555555555')
    monkeypatch.setenv('SLURM_LOCALID', '0')

    pinned = pin_device()
    assert not pinned.pinned
    assert not pinned.usable
    # is_mig comes from the matched record, so an unpinned rank cannot set it;
    # can_share is therefore NOT blocked by it. The warning is the only signal.
    assert pinned.is_mig is False
    message = ' '.join(pinned.warnings)
    assert 'MIG' in message
    assert 'not in the node device map' in message


def test_unmatched_device_uuid_mask_also_warns(monkeypatch):
    """Same miss for a plain GPU- UUID: the warning is not MIG-specific."""
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',
                       'GPU-99999999-9999-9999-9999-999999999999')
    monkeypatch.setenv('SLURM_LOCALID', '0')
    pinned = pin_device()
    assert not pinned.pinned
    assert any('not in the node device map' in w for w in pinned.warnings)


def test_unset_mask_permits_every_device(monkeypatch):
    """Unset means the launcher did not restrict anything."""
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setenv('SLURM_LOCALID', '1')
    pinned = pin_device()
    assert pinned.permitted == ('0', '1')
    assert pinned.pinned


@pytest.mark.parametrize('mask', ['', '   ', ',', ' , '])
def test_empty_mask_permits_no_device(monkeypatch, mask):
    """An EMPTY CUDA_VISIBLE_DEVICES is the opposite of an unset one.

    `CUDA_VISIBLE_DEVICES=""` is how a launcher says this rank gets no GPU,
    and CUDA reports zero devices for it. Both used to share a branch, so an
    empty mask expanded to every device on the node and the rank could select
    a card outside its allocation -- quietly sharing another job's GPU.
    """
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    monkeypatch.setenv('SLURM_LOCALID', '1')

    pinned = pin_device()
    assert pinned.permitted == ()
    assert not pinned.pinned
    assert not pinned.usable
    assert pinned.requested_pci == ''
    message = ' '.join(pinned.warnings)
    assert 'no GPU' in message
    assert 'CUDA_VISIBLE_DEVICES' in message


def test_empty_mask_does_not_borrow_a_device_from_the_node(monkeypatch):
    """The specific regression: with two devices present and an empty mask,
    no device may be selected at all."""
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    for local_rank in ('0', '1', '2'):
        monkeypatch.setenv('SLURM_LOCALID', local_rank)
        pinned = pin_device()
        assert pinned.requested_pci == '', \
            f'local rank {local_rank} selected {pinned.requested_pci} from an empty mask'


@pytest.mark.parametrize('mask', ['', '   ', ',', ' , '])
def test_empty_mask_is_recorded_as_a_flag(monkeypatch, mask):
    """Discovery needs to tell an empty mask from no GPU identity.

    Both leave `permitted == ()`, and re-reading CUDA_VISIBLE_DEVICES in
    discover_peers matched only the exact empty string, so ' ', ',' and ' , '
    still reached verify_pin and raised a raw CUDA error instead of the
    reason.
    """
    use_devices(monkeypatch, TWO_GPUS)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    monkeypatch.setenv('SLURM_LOCALID', '0')
    assert pin_device().empty_mask is True


def test_absent_identity_is_not_an_empty_mask(monkeypatch):
    """The no-identity branch also returns permitted == (), but the cause and
    the message differ, so the flag must not be set there."""
    monkeypatch.setattr(gp, 'node_devices', lambda: ({}, ''))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    pinned = pin_device()
    assert pinned.permitted == ()
    assert pinned.empty_mask is False
