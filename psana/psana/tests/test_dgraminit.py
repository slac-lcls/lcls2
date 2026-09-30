from psana import dgram
import os
import pytest

class TestDgramInit:

    @staticmethod
    def _jungfrau_test_file():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(
            dir_path,
            "test_data/detector/test_jungfrau05M_calib.xtc2",
        )

    def testInvalidEmptyDgram(self):
        raised = False
        try:
            d = dgram.Dgram()
        except RuntimeError:
            raised = True
        assert raised

    def testInvalidFileDescriptor(self):
        raised = False
        try:
            d = dgram.Dgram(file_descriptor=42)
        except:
            raised = True
        assert raised

    def testInvalidSequentialRead(self):
        """ prevent reading data dgram without config """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(dir_path, "smd.xtc2")
        if os.path.isfile(full_path):
            fd = os.open(full_path, os.O_RDONLY)
            d = dgram.Dgram(file_descriptor=fd)
            raised = False
            try:
                another_d = dgram.Dgram(file_descriptor=fd)
            except StopIteration:
                raised = True
            assert raised

    def testConfigureNamesMetadata(self):
        fd = os.open(self._jungfrau_test_file(), os.O_RDONLY)
        try:
            config = dgram.Dgram(file_descriptor=fd)
            entries = config.config_names()
        finally:
            os.close(fd)

        assert entries
        raw = next(
            entry
            for entry in entries
            if entry["det_name"] == "jungfrau"
            and entry["alg_name"] == "raw"
        )
        assert raw["det_type"] == "jungfrau"
        assert raw["alg_version"] == (0, 2, 0)
        assert raw["names_id_value"] == (
            (raw["node_id"] << 8) | raw["names_id"]
        )
        assert raw["n_fields"] == len(raw["fields"])

        fields = raw["fields"]
        assert [field["field_index"] for field in fields] == list(
            range(len(fields))
        )
        array_fields = [field for field in fields if field["rank"] > 0]
        assert [field["shape_index"] for field in array_fields] == list(
            range(len(array_fields))
        )
        assert all(
            field["shape_index"] == -1
            for field in fields
            if field["rank"] == 0
        )
        assert all(field["element_size"] > 0 for field in fields)

        raw_field = fields[0]
        assert raw_field == {
            "name": "raw",
            "type": 1,
            "element_size": 2,
            "rank": 3,
            "field_index": 0,
            "shape_index": 0,
        }

    def testConfigNamesRejectsEventDgram(self):
        fd = os.open(self._jungfrau_test_file(), os.O_RDONLY)
        try:
            config = dgram.Dgram(file_descriptor=fd)
        finally:
            os.close(fd)

        event = dgram.Dgram(config=config, fake_endrun=1)
        with pytest.raises(
            RuntimeError,
            match="only available on Configure dgrams",
        ):
            event.config_names()

def run():
    test = TestDgramInit()
    test.testInvalidEmptyDgram()
    test.testInvalidFileDescriptor()
    test.testInvalidSequentialRead()
    test.testConfigureNamesMetadata()
    test.testConfigNamesRejectsEventDgram()

if __name__ == "__main__":
    run()
