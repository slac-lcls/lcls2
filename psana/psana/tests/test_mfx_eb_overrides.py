import os

import pytest

from psana import datasource


@pytest.fixture(autouse=True)
def clean_mfx_override_env(monkeypatch):
    for name in (
        "PS_EB_NODES",
        "PS_EB_NODE_LOCAL",
        "PS_EB_PER_NODE",
        "PS_SMD_N_EVENTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(datasource, "_detect_node_count", lambda: 4)


def test_mfx_defaults_eb_nodes_to_detected_node_count():
    kwargs = {}

    datasource._force_mfx_overrides("mfx101572426", kwargs)

    assert os.environ["PS_EB_NODES"] == "4"
    assert os.environ["PS_SMD_N_EVENTS"] == "1000"
    assert kwargs["batch_size"] == 1


def test_mfx_preserves_explicit_nonlocal_eb_count(monkeypatch):
    monkeypatch.setenv("PS_EB_NODE_LOCAL", "0")
    monkeypatch.setenv("PS_EB_NODES", "12")

    datasource._force_mfx_overrides("mfx101572426", {})

    assert os.environ["PS_EB_NODES"] == "12"


def test_mfx_local_eb_count_uses_nodes_times_ebs_per_node(monkeypatch):
    monkeypatch.setenv("PS_EB_NODE_LOCAL", "1")
    monkeypatch.setenv("PS_EB_PER_NODE", "3")

    datasource._force_mfx_overrides("mfx101572426", {})
    datasource._ensure_local_eb_nodes()

    assert os.environ["PS_EB_NODES"] == "12"


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_mfx_invalid_explicit_eb_count_falls_back_to_node_count(
    monkeypatch, value
):
    monkeypatch.setenv("PS_EB_NODES", value)

    datasource._force_mfx_overrides("mfx101572426", {})

    assert os.environ["PS_EB_NODES"] == "4"
