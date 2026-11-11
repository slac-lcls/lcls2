import pytest

from psana.psexp.node import check_mpi_layout


@pytest.mark.parametrize(
    "world_size, ps_eb_nodes, ps_srv_nodes, expected",
    [
        (18, 3, 3, (3, 11)),  # 3 nodes x 6 tasks, the layout that works for TMO
        (10, 3, 3, (3, 3)),   # one BD per EB is the smallest valid layout
        (6, 1, 1, (1, 3)),    # the classic six core minimum
        (4, 1, 1, (1, 1)),
        (3, 1, 0, (1, 1)),
        (20, 2, 0, (2, 17)),
    ],
)
def test_valid_layouts(world_size, ps_eb_nodes, ps_srv_nodes, expected):
    assert check_mpi_layout(world_size, ps_eb_nodes, ps_srv_nodes) == expected


@pytest.mark.parametrize(
    "world_size, ps_eb_nodes, ps_srv_nodes",
    [
        (9, 3, 3),   # 3 nodes x 3 tasks: third EB gets no BD, job hangs
        (7, 3, 3),   # every non smd0 rank becomes an EB
        (5, 3, 1),
        (3, 1, 1),   # too few ranks for even one worker
        (6, 3, 0),   # 5 non smd0 ranks, 3 EBs, 2 BDs
    ],
)
def test_eb_without_bd_is_rejected(world_size, ps_eb_nodes, ps_srv_nodes):
    with pytest.raises(ValueError) as excinfo:
        check_mpi_layout(world_size, ps_eb_nodes, ps_srv_nodes)
    msg = str(excinfo.value)
    assert "BigData" in msg
    assert f"PS_EB_NODES:    {ps_eb_nodes}" in msg
    assert f"PS_SRV_NODES:   {ps_srv_nodes}" in msg
