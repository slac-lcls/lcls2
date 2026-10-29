from psdaq.slurm.config import Config


def test_main_config_ids_are_stripped():
    config = Config([{"id": " timing_0 ", "cmd": "drp"}])

    assert list(config.main_config) == ["timing_0"]


def test_select_finds_id_defined_with_whitespace():
    config = Config([{"id": "teb0 ", "cmd": "teb"}])
    config.select(["teb0"])

    assert list(config.select_config) == ["teb0"]


def test_select_and_add_strip_dict_ids():
    config = Config([])
    config.select([{"id": " control", "cmd": "control"}])
    config.add({"id": "daq ", "cmd": "daq"})

    assert sorted(config.select_config) == ["control", "daq"]
