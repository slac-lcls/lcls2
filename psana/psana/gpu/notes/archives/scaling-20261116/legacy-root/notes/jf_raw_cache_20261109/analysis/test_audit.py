import copy,unittest
from audit import gate

def record():
    files=[dict(path=str(i),pages=10,resident_pages=0) for i in range(5)]
    return dict(conditioning=dict(kind='cold',cache=dict(files=files,pages=50,resident_pages=0)),loop_seconds=2,nic_before=dict(p=dict(device='d',monotonic=1,rx_bytes_phy=0,tx_bytes_phy=0)),nic_after=dict(p=dict(device='d',monotonic=4,rx_bytes_phy=30,tx_bytes_phy=1)),nic_delta=dict(p=dict(seconds=3,rx_bytes=30,tx_bytes=1,rx_gib_s=30/(1<<30)/3)))
class Audit(unittest.TestCase):
    def test_valid(self):gate(record(),'cold')
    def test_cold(self):
        r=record();r['conditioning']['cache']['files'][0]['resident_pages']=1;r['conditioning']['cache']['resident_pages']=1
        with self.assertRaises(ValueError):gate(r,'cold')
    def test_warm_per_file(self):
        r=record();c=r['conditioning'];c['kind']='warm'
        for f in c['cache']['files']:f.update(pages=1000,resident_pages=1000)
        c['cache']['files'][0].update(pages=10,resident_pages=9)
        c['cache'].update(pages=4010,resident_pages=4009)
        self.assertGreater(c['cache']['resident_pages']/c['cache']['pages'],.999)
        with self.assertRaises(ValueError):gate(r,'warm')
    def test_warm_valid(self):
        r=record();c=r['conditioning'];c['kind']='warm'
        for f in c['cache']['files']:f['resident_pages']=f['pages']
        c['cache']['resident_pages']=c['cache']['pages'];gate(r,'warm')
    def test_wrong_kind(self):
        r=record();r['conditioning']['kind']='warm'
        with self.assertRaises(ValueError):gate(r,'cold')
    def test_reset(self):
        r=record();r['nic_after']['p']['rx_bytes_phy']=-1
        with self.assertRaises(ValueError):gate(r,'cold')
    def test_bracket(self):
        r=record();r['loop_seconds']=4
        with self.assertRaises(ValueError):gate(r,'cold')
    def test_duplicate_file(self):
        r=record();r['conditioning']['cache']['files'][0]['path']='1'
        with self.assertRaises(ValueError):gate(r,'cold')
    def test_bad_delta(self):
        r=record();r['nic_delta']['p']['rx_bytes']=31
        with self.assertRaises(ValueError):gate(r,'cold')

class Unavailable(unittest.TestCase):
    def test_failed_warm_has_no_rate(self):
        from audit import unavailable
        r=dict(status='cache-unavailable',settings=dict(cache='warm'))
        c=dict(status='fail',kind='warm',history=[{}]*9+[dict(stage='repair-8')],cache=dict(files=[dict(pages=10,resident_pages=9)]*5))
        unavailable(r,c)
        r['gib_s']=0
        with self.assertRaises(ValueError):unavailable(r,c)
    def test_passed_cache_cannot_be_missing(self):
        from audit import unavailable
        r=dict(status='cache-unavailable',settings=dict(cache='warm'))
        c=dict(status='fail',kind='warm',history=[{}]*9+[dict(stage='repair-8')],cache=dict(files=[dict(pages=10,resident_pages=10)]*5))
        with self.assertRaises(ValueError):unavailable(r,c)
