#
# psana script for LCLS1!!
# Assumes that typed_json is on the PYTHONPATH and json2xtc is on the path.
#
import sys
from typed_json import *
import subprocess

from psana import *
ds = DataSource('exp=xpptut15:run=54')
cfg = ds.env().configStore()

# the two fundamental EVR config objects we want to check
iocfg = cfg.get(EvrData.IOConfigV2, Source('ProcInfo()'))
evrcfg = cfg.get(EvrData.ConfigV7, Source('DetInfo(NoDetector.0:Evr.0)'))

"""
Two data types: EvrData.IOConfigV2 (Id_EvrIOCOnfig), EvrData.ConfigV7 (Id_EvrConfig).

EvrData.IOConfigV2:
   uint32_t _nchannels;
   EvrData::IOChannelV2	channels[this->_nchannels];
       EvrData::OutputMapV2	output;
           uint32_t _v;  (module(8) << 24) | (conn_id(8) << 16) | (conn(4) << 12) | (source_id(8) << 4) | source(4)
           source: pulse, dbus, prescaler, force_high, force_low
           conn: frontpanel, univio
       char	                name[NameLength=64];
       uint32_t	                ninfo;
       Pds::DetInfo	        infos[MaxInfos=16];
          uint32_t _log         (processId & 0x00ffffff) | (level(8) << 24)
          uint32_t _phy         Detector << 24, detId << 16, Device << 8, devId

EvrData.ConfigV7:
   uint32_t _neventcodes
   uint32_t _npulses;
   uint32_t _noutputs;
   EvrData::EventCodeV6	        eventcodes[this->_neventcodes];
      uint16_t code
      uint16_t maskeventattr (isReadout | (isCommand << 1) | (isLatch << 2))
      uint32_t reportDelay
      uint32_t reportWidth
      uint32_t maskTrigger
      uint32_t maskSet
      uint32_t maskClear
      char     *desc
      uint32_t readGroup
   EvrData::PulseConfigV3	pulses[this->_npulses];
      uint16_t pulseId
      uint16_t polarity
      uint32_t prescale
      uint32_t delay
      uint32_t width
   EvrData::OutputMapV2	        output_maps[this->_noutputs];
      uint32_t _v;  (module(8) << 24) | (conn_id(8) << 16) | (conn(4) << 12) | (source_id(8) << 4) | source(4)
      source: pulse, dbus, prescaler, force_high, force_low
      conn: frontpanel, univio
   EvrData::SequencerConfigV1	seq_config()
      uint32_t source  (beam_source(8) << 8 | sync_source(8)) # r120Hz,  r60Hz, r30Hz, r10Hz, r5Hz, r1Hz, r0_5Hz, Disable
      uint32_t length
      uint32_t cycles
      EvrData::SequencerEntry	data[this->length()];
          uint32_t value (eventcode(8) << 24 | delay(24))
"""

evrdev = cdict()
evrdev.setInfo("EvrConfig", "MyEvr", "", "")
evrdev.setAlg("raw", [0,0,0])

io = cdict()
io.set("nchannels", iocfg.nchannels())
for (i, ch) in enumerate(iocfg.channels()):
    chd = cdict()
    od = cdict()
    out = ch.output()
    od.set("module",  out.module(),     "UINT8")
    od.set("connid",  out.conn_id(),    "UINT8") # Deleted "_" in name!!
    od.set("conn",    int(out.conn()),  "UINT8") # 4-bit!
    od.set("source",  int(out.source()),"UINT8") # 4-bit!
    chd.set("output", od)                        # Deleted "_" in name!!
    chd.set("name", ch.name(), "CHARSTR")
    chd.set("ninfo", ch.ninfo())
    for infonum in range(ch.ninfo()):
        info = ch.infos()[infonum]
        id = cdict()
        id.set("detector", info.detector(), "UINT8")
        id.set("detid",    info.detId(),    "UINT8")
        id.set("device",   info.device(),   "UINT8")
        id.set("devid",    info.devId(),    "UINT8")
        id.set("level",    info.level(),    "UINT8")
        id.set("processId",info.processId(),"UINT32")
        chd.set("infos", id, append=True)
    io.set("channels", chd, append=True)

evrdev.set("IO", io)

evr = cdict()
evr.set("neventcodes", evrcfg.neventcodes())
for (i, ec) in enumerate(evrcfg.eventcodes()):
    ecd = cdict()
    ecd.set("code", ec.code(), "UINT16")
    ecd.set("maskeventattr", ec.isReadout() | (ec.isCommand() << 1) | (ec.isLatch() << 2), "UINT16")
    ecd.set("reportDelay",   ec.reportDelay(), "UINT32")
    ecd.set("reportWidth",   ec.reportWidth(), "UINT32")
    ecd.set("maskTrigger",   ec.maskTrigger(), "UINT32")
    ecd.set("maskSet",       ec.maskSet(),     "UINT32")
    ecd.set("maskClear",     ec.maskClear(),   "UINT32")
    ecd.set("desc",          ec.desc(),        "CHARSTR")
    ecd.set("readoutGroup",  ec.readoutGroup(),"UINT32")
    evr.set("eventcodes", ecd, append=True)
evr.set("npulses", evrcfg.npulses())
for (i, pls) in enumerate(evrcfg.pulses()):
    pld = cdict()
    pld.set("pulseId",  pls.pulseId(), "UINT16")
    pld.set("polarity", pls.polarity(),"UINT16")
    pld.set("prescale", pls.prescale(),"UINT32")
    pld.set("delay",    pls.width(),   "UINT32")
    pld.set("width",    pls.width(),   "UINT32")
    evr.set("pulses", pld, append=True)
evr.set("noutputs", evrcfg.noutputs())
for (i, out) in enumerate(evrcfg.output_maps()):
    od = cdict()
    od.set("module",  out.module(),     "UINT8")
    od.set("connid",  out.conn_id(),    "UINT8") # Deleted "_" in name!!
    od.set("conn",    int(out.conn()),  "UINT8") # 4-bit!
    od.set("source",  int(out.source()),"UINT8") # 4-bit!
    evr.set("outputmaps", od, append=True)       # Deleted "_" in name!!

seq = evrcfg.seq_config()
sd = cdict()
sd.set("source", (int(seq.beam_source()) << 8) | int(seq.sync_source()), "UINT8")
sd.set("cycles", seq.cycles(), "UINT32")
sd.set("length", seq.length())
for (i, se) in enumerate(seq.entries()):
    sed = cdict()
    sed.set("eventcode", se.eventcode(), "UINT8")
    sed.set("delay",     se.delay(),     "UINT32")  # Actually 24.
evr.set("seqconfig", seq)

evrdev.set("evr", evr)

evrdev.writeFile("test_evr.json")
subprocess.call(["json2xtc", "test_evr.json", "test_evr.xtc2"])
