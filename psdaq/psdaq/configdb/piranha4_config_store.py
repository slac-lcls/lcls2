from psdaq.configdb.typed_json import cdict
from psdaq.configdb.tsdef import *
import psdaq.configdb.configdb as cdb
import sys
import IPython
import argparse

def piranha4_cdict():

    #database contains collections which are sets of documents (aka json objects).
    #each type of device has a collection.  The elements of that collection are configurations of that type of device.
    #e.g. there will be OPAL, EVR, and JUNGFRAU will be collections.  How they are configured will be a document contained within that collection
    #Each hutch is also a collection.  Documents contained within these collection have an index, alias, and list of devices with configuration IDs
    #How is the configuration of a state is described by the hutch and the alias found.  E.g. TMO and BEAM.  TMO is a collection.
    #BEAM is an alias of some of the documents in that collection. The document with the matching alias and largest index is the current
    #configuration for that hutch and alias.
    #When a device is configured, the device has a unique name OPAL7.  Need to search through document for one that has an NAME called OPAL7.  This will have
    #have two fields "collection" and ID field (note how collection here is a field. ID points to a unique document).  This collection field and
    #ID point to the actuall Mongo DB collection and document

    top = cdict()
    top.setAlg('config', [2,2,0])

    top.set("firmwareBuild:RO"  , "-", 'CHARSTR')
    top.set("firmwareVersion:RO",   0, 'UINT32')

    help_str  = "-- user interface --"
    help_str += "\nstart_ns     : nanoseconds from fiducial to exposure start"
    help_str += "\ngate_ns      : nanoseconds of exposure; 25 ns resolution"
    help_str += "\nblack_level  : contrast offset; range is dependent on pixel format"
    help_str += "\nvertical_bin : single pixel or binning of 2 pixels"
    top.set("help:RO", help_str, 'CHARSTR')

    top.define_enum('clmEnum',   {'Base':0, 'Medium':1, 'Full':2})
    top.define_enum('clsEnum',   {'85MHz':0, '66MHz':1})
    top.define_enum('dstEnum',   {'Line':0, 'Area':1})
    top.define_enum('binEnum',   {'1_pixel':1, '2_pixels':2})
    top.define_enum('dirEnum',   {'Forward':0, 'Reverse':1, 'External':2})
    top.define_enum('expEnum',   {'Internal':0, 'External':1})
    top.define_enum('offOnEnum', {'Off':0, 'On':1})
    top.define_enum('pixEnum',   {'8_bits':0, '10_bits':1, '12_bits':2})
    top.define_enum('tpEnum',    {'Sensor_Video':0, 'Ramp':1, 'A5':3, 'Each_tap_fixed':4, 'All_1365':5, 'All_1':6})
    top.define_enum('tdiEnum',   {'Single_line':1, 'TDI':2})
    top.define_enum('sesEnum',   {'All':0, 'Bottom':1, 'Top':2})
    top.define_enum('usdEnum',   {'Factory':0})
    top.define_enum('ffmEnum',   {'Disable':0, 'Enable':1, 'Reset':2, 'Scan':3})

    #Create a user interface that is an abstraction of the common inputs
    top.set("user.start_ns",  87250, 'UINT32')
    top.set("user.gate_ns" , 100000, 'UINT32')
    top.set("user.black_level",  16,  'INT32')
    top.set("user.vertical_bin",  2, 'binEnum')

    # timing system
    top.set('expert.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.PauseThreshold',16,'UINT32')
    top.set('expert.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.TriggerDelay',42,'UINT32')
    top.set('expert.ClinkPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer.Partition:RO',0,'UINT32')

    top.define_enum('rateEnum', fixedRateHzToMarker)
    top.set('expert.ClinkPcie.Hsio.TimingRx.XpmMiniWrapper.XpmMini.Config_L0Select_RateSel',6,'rateEnum')

    # Feb[0] refers to pgp lane, Ch[0],[1] refers to camera link channel from Feb (these should be abstracted)
    # UartPiranha4 is camType; sets serial registers
    # ClinkTop.LinkMode is Disable,Base,Medium,Full
    # ClinkTop.DataMode is None,8b,10b,12b
    # ClinkTop.FrameMode is None,Line,Frame
    # ClinkTop.TapCount
    # All serial commands are enumerated as registers
    top.set('expert.ClinkFeb.TrigCtrl.EnableTrig', 1, 'UINT8')   # rogue wants 'bool'
    top.set('expert.ClinkFeb.TrigCtrl.InvCC'     , 0, 'UINT8')   # rogue wants 'bool'
    top.set('expert.ClinkFeb.TrigCtrl.TrigMap'   , 0, 'UINT32')  # ChanA/ChanB
    top.set('expert.ClinkFeb.TrigCtrl.TrigMask'  , 1, 'UINT32')  # CC1
    top.set('expert.ClinkFeb.TrigCtrl.TrigPulseWidth', 4.000, 'FLOAT')  #32.768, 'FLOAT')

    top.set("expert.ClinkFeb.ClinkTop.PllConfig0"      ,'85MHz','CHARSTR')
    top.set("expert.ClinkFeb.ClinkTop.PllConfig1"      ,'85MHz','CHARSTR')
    top.set("expert.ClinkFeb.ClinkTop.PllConfig2"      ,'85MHz','CHARSTR')

    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.LinkMode"    ,      2,'UINT32')   # Medium mode
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.DataMode"    ,      3,'UINT32')   # 12-bit: Requires Base or Medium LinkMode
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.FrameMode"    ,     1,'UINT32')   # Line
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.TapCount"    ,      4,'UINT32')   #
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.DataEn"    ,        1,'UINT8')    # rogue wants 'bool'
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.Blowoff"    ,       0,'UINT8')    # rogue wants 'bool'
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.BaudRate"      , 9600,'UINT32')   # bps
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.SerThrottle"   ,30000,'UINT32')
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.SwControlValue",    0,'UINT32')   # Frame
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.SwControlEn"   ,    0,'UINT32')   # Frame

    # The following are mostly in the order of the GCP dump
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SEM",        0,'expEnum')   # Internal Exposure Mode for quicker commanding (?!)
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.USD",        0,'usdEnum')   # Default User Set <0:Factory>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.STM",        1,'expEnum')   # Trigger Mode <0:Int 1:Ext>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SSF",        1, 'UINT32')   # Internal Line Rate <Hz>; Requires 'STM 0'
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SES",        0,'sesEnum')   # Exposure Selector <0:All, 1:Bottom 2:Top>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SME",        0,'offOnEnum') # Multi-Exposure <0:Off, 1:On>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SET",     4000, 'UINT32')   # Exposure Time <ns>; Min value is 4 us; Requires 'SEM 0'

    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SVM",        0,'tpEnum')    # Test Pattern <0:Off, 1:Ramp, 3-6>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SCD",        0,'dirEnum')   # Direction <0:Fwd, 1:Rev 2:Ext>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.STG",        1,'tdiEnum')   # Set TDI Stages <1:Single line, 2:TDI>       [Must preceed DST command]
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SBV",        1,'binEnum')   # Vertical Binning <1|2> pixels
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SBH",        1,'binEnum')   # Horizontal Binning <1|2> pixels
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.FFM",        0,'ffmEnum')   # Flat Field Mode <0:Disable, 1:Enable, 2:Reset, 3:Scan>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SSB",        0,  'INT32')   # Contrast Offset <DN>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SSG", '0 f1.0','CHARSTR')   # Gain <0:System, 1:Bottom Line, 2:Top Line>  f<gain>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SMM",        0,'offOnEnum') # Mirroring <0:Off 1:On>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SAM",        0,'offOnEnum') # AOI Mode <0:Off/Disable 1:Active/Enable>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.DST",        0,'dstEnum')   # Device Scan Type <0:Line Scan, 1:Area Scan> [Must preceed SBV command]
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.CLS",        0,'clsEnum')   # Camera Link Speed  <0:85MHz, 1:66MHz>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.CLM",        1,'clmEnum')   # Camera Link Mode <0:Base 1:Med 2:Full>
    top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.SPF",        2,'pixEnum')   # Pixel Format <0:8 bits 1:10 bits 2:12 bits>
    # This is broken or I'm not doing it right:
    #top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.ROI[0]",     1, 'UINT32')   # Flat field ROI first pixel <An int in range [1-2048], 1st < 2nd>
    #top.set("expert.ClinkFeb.ClinkTop.ClinkCh.UartPiranha4.ROI[1]",  2048, 'UINT32')   # Flat field ROI last  pixel <An int in range [1-2048], 2nd > 1st>

    return top

if __name__ == "__main__":
    create = True
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

    args = cdb.createArgs().args

    # These override the command line arguments, so left commented out
    #args.name = 'tstpiranha4'
    #args.segm = 0
    #args.id = 'piranha4_serial1235'
    #args.alias = 'BEAM'
    #args.prod = True
    #args.inst = 'tst'
    #args.user = 'tstopr'
    #args.password = 'pcds'

    db   = 'configdb' if args.prod else 'devconfigdb'
    url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

    mycdb = cdb.configdb(url, args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config('piranha4')

    top = piranha4_cdict()
    top.setInfo('piranha4', args.name, args.segm, args.id, 'No comment')

    retval = mycdb.modify_device(args.alias, top)
    if retval == 0:
        print('Failed to store configuration')
        sys.exit(1)
