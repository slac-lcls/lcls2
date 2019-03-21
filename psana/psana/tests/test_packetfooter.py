from psana.psexp.packet_footer import PacketFooter
import unittest

class TestPacketFooter(unittest.TestCase) :

    def test_contents(self):
        view = bytearray()
        pf = PacketFooter(2)
        for i, msg in enumerate([b'packet0', b'packet1']):
            view.extend(msg)
            pf.set_size(i, memoryview(msg).shape[0])

        view.extend(pf.footer)

        pf2 = PacketFooter(view=view)
        assert pf2.n_packets == 2

        views = pf2.split_packets()
        assert memoryview(views[0]).shape[0] == 7
        assert memoryview(views[1]).shape[0] == 7


if __name__ == "__main__":
    unittest.main()
