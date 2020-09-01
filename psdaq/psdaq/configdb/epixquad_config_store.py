from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import numpy as np
import sys
import IPython
import argparse

def epixquad_cdict(args):

    #database contains collections which are sets of documents (aka json objects).
    #each type of device has a collection.  The elements of that collection are configurations of that type of device.
    #e.g. there will be EPIXQUAD, EVR, and YUNGFRAU will be collections.  How they are configured will be a document contained within that collection
    #Each hutch is also a collection.  Documents contained within these collection have an index, alias, and list of devices with configuration IDs
    #How is the configuration of a state is described by the hutch and the alias found.  E.g. TMO and BEAM.  TMO is a collection.
    #BEAM is an alias of some of the documents in that collection. The document with the matching alias and largest index is the current
    #configuration for that hutch and alias.
    #When a device is configured, the device has a unique name EPIXQUAD7.  Need to search through document for one that has an NAME called EPIXQUAD7.  This will have
    #have two fields "collection" and ID field (note how collection here is a field. ID points to a unique document).  This collection field and
    #ID point to the actuall Mongo DB collection and document

    top = cdict()
    top.setInfo('epixquad', args.name, args.segm, args.id, 'No comment')
    top.setAlg('config', [2,0,0])

    #top.set("firmwareBuild:RO"  , "-", 'CHARSTR')
    #top.set("firmwareVersion:RO",   0, 'UINT32')

    help_str  = "-- user interface --"
    help_str += "\nstart_ns           : nanoseconds from fiducial to exposure start"
    help_str += "\ngain.mode          : High/Medium/Low/AutoHiLo/AutoMedLo/Map"
    help_str += "\ngain.map           : 3D-map of pixel gain settings"
    help_str += "\n  -- charge injection --"
    help_str += "\ncharge_inj.mode    : Disable/Map/Square(spacing)"
    help_str += "\ncharge_inj.map     : 3D-map of charge injection settings (Map mode)"
    help_str += "\ncharge_inj.square_spacing : Pixels between injection sites (Square mode)"
    help_str += "\ncharge_inj.square_site    : Pixel index [0..spacing^2] (Square mode)"
    top.set("help:RO", help_str, 'CHARSTR')

    top.define_enum('boolEnum', {'False':0, 'True':1})
    top.define_enum('gainEnum', {'High':0, 'Medium':1, 'Low':2, 'AutoHiLo':3, 'AutoMedLo':4, 'Map':5})
    top.define_enum('chargeModeEnum', {'Disable':0, 'Map':1, 'Square':2})

    pixelMap = []
    for ia in range(16):
        pixelMap.append(np.zeros((178, 192),dtype=np.uint8))

    #Create a user interface that is an abstraction of the common inputs
    top.set("user.start_ns" , 107749, 'UINT32')
    top.set("user.gain_mode",      0, 'gainEnum')  
    top.set("user.gain_map", pixelMap, 'UINT8')  

    top.set("user.charge_inj.mode",    0, 'chargeModeEnum')
    top.set("user.charge_inj.map", pixelMap, 'UINT8')
    top.set("user.charge_inj.square_spacing", 7, 'UINT8')
    top.set("user.charge_inj.square_site"   , 0, 'UINT16')

    # timing system
    top.set('expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].PauseThreshold',16,'UINT32')
    top.set('expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].TriggerDelay',42,'UINT32')
    top.set('expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].Partition',0,'UINT32')

    top.define_enum('rateEnum', {'929kHz':0, '71kHz':1, '10kHz':2, '1kHz':3, '100Hz':4, '10Hz':5, '1Hz':6})
    top.set('expert.DevPcie.Hsio.TimingRx.XpmMiniWrapper.XpmMini.Config_L0Select_RateSel',6,'rateEnum')

    top.define_enum('trigSrcEnum', {'0':0, '1':1, '2':2, '3':3 })
    base = 'expert.EpixQuad.SystemRegs.'
    top.set(base+'AutoTrigEn' , 0, 'boolEnum')
    top.set(base+'AutoTrigPer', 270000, 'UINT32')
    top.set(base+'DcDcEnable' , 0xf, 'UINT8')
    top.set(base+'AsicAnaEn'  , 0, 'boolEnum')
    top.set(base+'AsicDigEn'  , 0, 'boolEnum')
    top.set(base+'DdrVttEn'   , 0, 'boolEnum')
    top.set(base+'TrigSrcSel' , 0, 'trigSrcEnum')
    top.set(base+'VguardDac'  , 0, 'UINT16')

    base = 'expert.EpixQuad.AcqCore.'
    top.set(base+'AcqToAsicR0Delay', 0, 'UINT32')
    top.set(base+'AsicR0Width', 100, 'UINT32')
    top.set(base+'AsicR0ToAsicAcq', 100, 'UINT32')
    top.set(base+'AsicAcqWidth', 100, 'UINT32')
    top.set(base+'AsicAcqLToPpmatL', 0, 'UINT32')
    top.set(base+'AsicPpmatToReadout', 0, 'UINT32')
    top.set(base+'AsicRoClkHalfT', 3, 'UINT32')
    top.set(base+'AsicAcqForce', 0, 'boolEnum')
    top.set(base+'AsicAcqValue', 0, 'boolEnum')
    top.set(base+'AsicR0Force' , 0, 'boolEnum')
    top.set(base+'AsicR0Value' , 0, 'boolEnum')
    top.set(base+'AsicPpmatForce' , 1, 'boolEnum')
    top.set(base+'AsicPpmatValue' , 1, 'boolEnum')
    top.set(base+'AsicSyncForce' , 0, 'boolEnum')
    top.set(base+'AsicSyncValue', 0, 'boolEnum')
    top.set(base+'AsicRoClkForce', 0, 'boolEnum')
    top.set(base+'AsicRoClkValue', 0, 'boolEnum')

    base = 'expert.EpixQuad.RdoutCore.'
    top.set(base+'AdcPipelineDelay', 0, 'UINT32')
    top.set(base+'TestData', 0, 'boolEnum')

    base = 'expert.EpixQuad.PseudoScopeCore.'
    top.set(base+'ScopeEn', 0, 'boolEnum')
    top.set(base+'TrigEdge', 0, 'boolEnum')
    top.set(base+'TrigChannel', 0, 'UINT8')
    top.set(base+'TrigMode', 0, 'UINT8')
    top.set(base+'TrigAdcThreshold', 0, 'UINT16')
    top.set(base+'TrigHoldoff', 0, 'UINT16')
    top.set(base+'TrigOffset', 0, 'UINT16')
    top.set(base+'TrigDelay', 0, 'UINT16')
    top.set(base+'TraceLength', 0, 'UINT16')
    top.set(base+'SkipSamples', 0, 'UINT16')
    top.set(base+'InChannelA', 0, 'UINT8')
    top.set(base+'InChannelB', 0, 'UINT8')

    base = 'expert.EpixQuad.VguardDac.'
    top.set(base+'VguardDacRaw', 0, 'UINT16')

    for i in range(16):
        base = 'expert.EpixQuad.Epix10kaSaci[{}]'.format(i)
        #top.set(base+'REG', 0, 'UINT32')

    #top.set('expert.EpixQuad.SaciConfigCore.', 0, 'UINT32')

    top.defineEnum('ExtPwdnEnum',{'FullPowerDown':0,'Standby':1})
    top.defineEnum('IntPwdnEnum',{'ChipRun':0, 'FullPowerDown':1, 'Standby':2, 'DigitalReset':3 })
    top.defineEnum('UserTestModeEnum',{ 'single':0, 'alternate':1, 'single_once':2, 'alternate_once':3 })
    top.defineEnum('OutputTestModeEnum', { 'Off':0, 'MidscaleShort':1, 'PosFS':2, 'NegFS':3, 'AltCheckerBoard':4, 'PN23':5, 'PN9':6, 'OneZeroWordToggle':7, 'UserInput':8, 'OneZeroBitToggle':9 })
    top.defineEnum('ClockDivideChEnum',{ '{}'.format(i+1):i for i in range(8)})
    top.defineEnum('OutputFormatEnum', { 'TwosComplement':0, 'OffsetBinary':1})

    for i in range(10):
        base = 'expert.EpixQuad.Ad9249Readout[{}]'.format(i)
        top.set(base+'FrameDelay', 160, 'UINT16')
        top.set(base+'Invert'    , 0, 'boolEnum')
        for j in range(8):
            cbase = base+'ChannelDelay[{}]'.format(j)
            top.set(cbase, 160, 'UINT16')

        base = 'expert.EpixQuad.Ad9249Config[{}]'.format(i)
        top.set(base+'ChipId:RO', '', 'CHARSTR')
        top.set(base+'ExternalPwdnMode', 0, 'ExtPwdnEnum')
        top.set(base+'InternalPwdnMode', 0, 'IntPwdnEnum')
        top.set(base+'DutyCycleStabilizer', 1, 'UINT8')
        top.set(base+'ClockDivide', 0, 'ClockDivideChEnum')
        top.set(base+'ChopMode', 0, 'UINT8')
        top.set(base+'DevIndexMask[7:0]', 0xff, 'UINT8')
        top.set(base+'DevIndexMask[DCO:FCO]', 0x3, 'UINT8')
        top.set(base+'UserTestModeCfg', 0, 'UserTestModeEnum')
        top.set(base+'OffsetAdjust', 0, 'UINT8')
        top.set(base+'OutputInvert', 0, 'UINT8')
        top.set(base+'OutputFormat', 0, 'OutputFormatEnum')

    base = 'expert.EpixQuad.Ad9249Tester.'
    top.set(base+'TestChannel', 0, 'UINT32')
    top.set(base+'TestDataMask', 0, 'UINT32')
    top.set(base+'TestPattern', 0, 'UINT32')
    top.set(base+'TestSamples', 0, 'UINT32')
    top.set(base+'TestTimeout', 0, 'UINT32')
    top.set(base+'TestRequest', 0, 'boolEnum')

    return top

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Write a new TimeTool configuration into the database')
    parser.add_argument('--inst', help='instrument', type=str, default='tst')
    parser.add_argument('--alias', help='alias name', type=str, default='BEAM')
    parser.add_argument('--name', help='detector name', type=str, default='tsttt')
    parser.add_argument('--segm', help='detector segment', type=int, default=0)
    parser.add_argument('--id', help='device id/serial num', type=str, default='serial1234')
    parser.add_argument('--user', help='user for HTTP authentication', type=str, default='xppopr')
    parser.add_argument('--password', help='password for HTTP authentication', type=str, default='pcds')
    args = parser.parse_args()

    create = True
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

    mycdb = cdb.configdb('https://pswww.slac.stanford.edu/ws-auth/devconfigdb/ws/', args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config('epixquad')

    top = epixquad_cdict(args)

    mycdb.modify_device(args.alias, top)
