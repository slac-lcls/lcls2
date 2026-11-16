import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mfx_kvikio_read_20261108'))
from network_watch import counter_delta

class Counters(unittest.TestCase):
    def setUp(self):
        self.a={'nic0':dict(monotonic=1,device='pci0',rx_bytes_phy=100,tx_bytes_phy=10)}
        self.b={'nic0':dict(monotonic=3,device='pci0',rx_bytes_phy=1100,tx_bytes_phy=30)}
    def test_bytes_and_interval(self):
        d=counter_delta(self.a,self.b,500)['nic0']
        self.assertEqual((d['rx_bytes'],d['seconds'],d['rx_to_requested_bytes']),(1000,2,2))
    def test_reset(self):
        self.b['nic0']['rx_bytes_phy']=0
        with self.assertRaises(ValueError): counter_delta(self.a,self.b)
    def test_missing(self):
        with self.assertRaises(ValueError): counter_delta(self.a,{})
    def test_clock(self):
        self.b['nic0']['monotonic']=1
        with self.assertRaises(ValueError): counter_delta(self.a,self.b)
    def test_ports_separate(self):
        a=copy.deepcopy(self.a);b=copy.deepcopy(self.b)
        a['nic1']=dict(a['nic0'],device='pci1');b['nic1']=dict(b['nic0'],device='pci1')
        self.assertEqual(set(counter_delta(a,b)),{'nic0','nic1'})
