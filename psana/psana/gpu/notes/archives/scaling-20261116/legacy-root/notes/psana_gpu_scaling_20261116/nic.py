"""Read-only Cassini hardware octet counters; no IP-only fallback or resets."""
import time
from pathlib import Path
from support import require

COUNTERS=('hni_sts_rx_ok_octets','hni_sts_tx_ok_octets')

def snapshot(root=Path('/sys/class/cxi')):
    started=time.time();devices={}
    for device in sorted(root.glob('cxi*')):
        pci=str((device/'device').resolve());values={}
        for name in COUNTERS:
            path=device/'device/telemetry'/name
            raw=path.read_text().strip();value,stamp=raw.split('@')
            require(value.isdecimal(),'Invalid hardware octet counter')
            values[name]={'value':int(value),'kernel_timestamp':float(stamp),'raw':raw,'path':str(path)}
        devices[device.name]={'pci':pci,'counters':values}
    require(devices and len({v['pci'] for v in devices.values()})==len(devices),'Missing or duplicate physical CXI devices')
    return {'started_unix':started,'finished_unix':time.time(),'devices':devices,'unit':'octets (bytes)','scope':'whole physical NIC; includes all traffic, not Lustre-attributed'}

def delta(before,after):
    require(before['devices'].keys()==after['devices'].keys() and before['finished_unix']<=after['started_unix'],'NIC device set/time changed')
    result={}
    for name,b in before['devices'].items():
        a=after['devices'][name];require(b['pci']==a['pci'],'NIC identity changed');r={}
        for counter in COUNTERS:
            first=b['counters'][counter];last=a['counters'][counter]
            require(last['value']>=first['value'] and last['kernel_timestamp']>first['kernel_timestamp'],'NIC reset/wrap/nonmonotonic sample')
            r[counter]=last['value']-first['value']
        result[name]=r
    return {'devices':result,'rx_bytes':sum(r[COUNTERS[0]] for r in result.values()),'tx_bytes':sum(r[COUNTERS[1]] for r in result.values()),'scope':'whole-NIC good octets; no storage-only attribution'}
