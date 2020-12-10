from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import numpy as np
import sys
import IPython
import argparse

def epixquad_cdict():

    top = cdict()
    top.setAlg('config', [2,0,0])

    #top.set("firmwareBuild:RO"  , "-", 'CHARSTR')
    #top.set("firmwareVersion:RO",   0, 'UINT32')

    help_str  = "-- user interface --"
    help_str += "\nstart_ns     : nanoseconds to exposure start"
    help_str += "\ngain_mode    : High/Med/Low/AutoHiLo/AutoMedLo/Map"
    help_str += "\npixel_map    : 3D-map of pixel gain/inj settings"
    #top.set("help.user:RO", help_str, 'CHARSTR')

    pixelMap = np.zeros((16,178,192),dtype=np.uint8)
    top.set("user.pixel_map", pixelMap)

    top.set("user.start_ns" , 107749, 'UINT32')    # segment 0 only

    top.define_enum('gainEnum', {'High':0, 'Medium':1, 'Low':2, 'AutoHiLo':3, 'AutoMedLo':4, 'Map':5})
    top.set("user.gain_mode",      0, 'gainEnum')  

    # timing system
    top.set('expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].PauseThreshold',16,'UINT32')
    top.set('expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].TriggerDelay',42,'UINT32')
    top.set('expert.DevPcie.Hsio.TimingRx.TriggerEventManager.TriggerEventBuffer[0].Partition',0,'UINT32')

    top.define_enum('rateEnum', {'929kHz':0, '71kHz':1, '10kHz':2, '1kHz':3, '100Hz':4, '10Hz':5, '1Hz':6})
    top.set('expert.DevPcie.Hsio.TimingRx.XpmMiniWrapper.XpmMini.Config_L0Select_RateSel',6,'rateEnum')

    top.define_enum('boolEnum', {'False':0, 'True':1})
    top.define_enum('trigSrcEnum', {'PGP':0, 'TTL':1, 'Cmd':2, 'Auto':3 })
    base = 'expert.EpixQuad.SystemRegs.'
    top.set(base+'AutoTrigEn' , 0, 'boolEnum')
    top.set(base+'AutoTrigPer', 270000, 'UINT32')
    top.set(base+'DcDcEnable' , 0xf, 'UINT8')
    top.set(base+'AsicAnaEn'  , 1, 'boolEnum')
    top.set(base+'AsicDigEn'  , 1, 'boolEnum')
    top.set(base+'DdrVttEn'   , 0, 'boolEnum')
    top.set(base+'TrigSrcSel' , 0, 'trigSrcEnum')
    top.set(base+'TrigEn'     , 1, 'boolEnum')
    top.set(base+'AsicMask'   , 0xffff, 'UINT16')

    base = 'expert.EpixQuad.AcqCore.'
    top.set(base+'AcqToAsicR0Delay', 0, 'UINT32')
    top.set(base+'AsicR0Width', 0x1e, 'UINT32')
    top.set(base+'AsicR0ToAsicAcq', 0x2710, 'UINT32')
    top.set(base+'AsicAcqWidth', 0x2710, 'UINT32')
    top.set(base+'AsicAcqLToPPmatL', 0x3e8, 'UINT32')
    top.set(base+'AsicPpmatToReadout', 0x0, 'UINT32')
    top.set(base+'AsicRoClkHalfT', 0x3, 'UINT32')
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
    top.set(base+'AsicSyncInjEn', 1, 'boolEnum')
    top.set(base+'AsicSyncInjDly', 0x3e8, 'UINT32')
    top.set(base+'DbgOutSel[0]', 0, 'UINT32')
    top.set(base+'DbgOutSel[1]', 0, 'UINT32')
    top.set(base+'DbgOutSel[2]', 0, 'UINT32')
    top.set(base+'DummyAcqEn', 0, 'boolEnum')  # ghost correction?

    base = 'expert.EpixQuad.RdoutCore.'
    top.set(base+'RdoutEn', 1, 'boolEnum')
    top.set(base+'AdcPipelineDelay', 0x44, 'UINT32')
    top.set(base+'TestData', 0, 'boolEnum')
    top.set(base+'OverSampleEn', 0, 'boolEnum')
    top.set(base+'OverSampleSize', 0, 'UINT8')

    top.define_enum('scopeTrigMode', {'Disable':0, 'Axil':1, 'Trig':2 })
    base = 'expert.EpixQuad.PseudoScopeCore.'
    top.set(base+'ScopeEn', 0, 'boolEnum')
    top.set(base+'TrigEdge', 0, 'boolEnum')
    top.set(base+'TrigChannel', 4, 'UINT8')
    top.set(base+'TrigMode', 2, 'UINT8')
    top.set(base+'TrigAdcThreshold', 0, 'UINT16')
    top.set(base+'TrigHoldoff', 0, 'UINT16')
    top.set(base+'TrigOffset', 0x46a, 'UINT16')
    top.set(base+'TrigDelay', 0, 'UINT16')
    top.set(base+'TraceLength', 0x1f40, 'UINT16')
    top.set(base+'SkipSamples', 0, 'UINT16')
    top.set(base+'InChannelA', 0x10, 'UINT8')
    top.set(base+'InChannelB', 0x11, 'UINT8')

    base = 'expert.EpixQuad.VguardDac.'
    top.set(base+'VguardDacRaw', 0, 'UINT16')

    for i in range(16):
        base = 'expert.EpixQuad.Epix10kaSaci[{}].'.format(i)
        top.set(base+'CompTH_DAC', 0x22, 'UINT8')
        top.set(base+'CompEn0', 0, 'UINT8')
        top.set(base+'CompEn1', 1, 'UINT8')
        top.set(base+'CompEn2', 1, 'UINT8')
        top.set(base+'PulserSync', 1, 'boolEnum')
        top.set(base+'PixelDummy', 0x22, 'UINT8')
        top.set(base+'Pulser', 0, 'UINT16')
        top.set(base+'pbit', 0, 'boolEnum')
        top.set(base+'atest', 0, 'boolEnum')
        top.set(base+'test', 0, 'boolEnum')
        top.set(base+'sab_test', 0, 'boolEnum')
        top.set(base+'hrtest', 0, 'boolEnum')
        top.set(base+'PulserR', 0, 'boolEnum')
        top.set(base+'DigMon1', 0, 'UINT8')
        top.set(base+'DigMon2', 1, 'UINT8')
        top.set(base+'PulserDac', 3, 'UINT8')
        top.set(base+'MonostPulser', 0, 'UINT8')
        top.set(base+'Dm1En', 0, 'boolEnum')
        top.set(base+'Dm2En', 0, 'boolEnum')
        top.set(base+'emph_bd', 0, 'UINT8')
        top.set(base+'emph_bc', 0, 'UINT8')
        top.set(base+'VRef', 0x13, 'UINT8')
        top.set(base+'VRefLow', 0x3, 'UINT8')
        top.set(base+'TpsTComp', 1, 'boolEnum')
        top.set(base+'TpsMux', 0, 'UINT8')
        top.set(base+'RoMonost', 0x3, 'UINT8')
        top.set(base+'TpsGr', 0x3, 'UINT8')
        top.set(base+'S2d0Gr', 0x3, 'UINT8')
        top.set(base+'PpOcbS2d', 1, 'boolEnum')
        top.set(base+'Ocb', 0x3, 'UINT8')
        top.set(base+'Monost', 0x3, 'UINT8')
        top.set(base+'FastppEnable', 0, 'boolEnum')
        top.set(base+'Preamp', 0x4, 'UINT8')
        top.set(base+'PixelCb', 0x4, 'UINT8')
        top.set(base+'Vld1_b', 0x1, 'UINT8')
        top.set(base+'S2dTComp', 0, 'boolEnum')
        top.set(base+'FilterDac', 0x11, 'UINT8')
        top.set(base+'TestLVDTransmitter', 0, 'boolEnum')
        top.set(base+'TC', 0, 'UINT8')
        top.set(base+'S2d', 0x3, 'UINT8')
        top.set(base+'S2dDacBias', 0x3, 'UINT8')
        top.set(base+'TpsTcDac', 0, 'UINT8')
        top.set(base+'TpsDac', 0x10, 'UINT8')
        top.set(base+'S2d0TcDac', 0x1, 'UINT8')
        top.set(base+'S2d0Dac', 0x14, 'UINT8')
        top.set(base+'TestBe', 0, 'boolEnum')
        top.set(base+'IsEn', 0, 'boolEnum')
        top.set(base+'DelExec', 0, 'boolEnum')
        top.set(base+'DelCckRef', 0, 'boolEnum')
        top.set(base+'RO_rst_en', 1, 'boolEnum')
        top.set(base+'SlvdsBit', 1, 'boolEnum')
        top.set(base+'FELmode', 1, 'boolEnum')
        top.set(base+'CompEnOn', 0, 'boolEnum')
        top.set(base+'RowStartAddr', 0, 'UINT16')
        top.set(base+'RowStopAddr', 0xb1, 'UINT16')
        top.set(base+'ColStartAddr', 0, 'UINT16')
        top.set(base+'ColStopAddr', 0x2f, 'UINT16')
        top.set(base+'S2d1Gr', 0x3, 'UINT8')
        top.set(base+'S2d2Gr', 0x3, 'UINT8')
        top.set(base+'S2d3Gr', 0x3, 'UINT8')
        top.set(base+'trbit', 0, 'boolEnum')
        top.set(base+'S2d1TcDac', 0x1, 'UINT8')
        top.set(base+'S2d1Dac', 0x12, 'UINT8')
        top.set(base+'S2d2TcDac', 0x1, 'UINT8')
        top.set(base+'S2d2Dac', 0x12, 'UINT8')
        top.set(base+'S2d3TcDac', 0x1, 'UINT8')
        top.set(base+'S2d3Dac', 0x12, 'UINT8')

    #top.set('expert.EpixQuad.SaciConfigCore.', 0, 'UINT32')

    top.define_enum('ExtPwdnEnum',{'FullPowerDown':0,'Standby':1})
    top.define_enum('IntPwdnEnum',{'ChipRun':0, 'FullPowerDown':1, 'Standby':2, 'DigitalReset':3 })
    top.define_enum('OffOnEnum',{'Off':0, 'On':1})
    top.define_enum('UserTestModeEnum',{ 'single':0, 'alternate':1, 'single_once':2, 'alternate_once':3 })
    top.define_enum('OutputTestModeEnum', { 'Off':0, 'MidscaleShort':1, 'PosFS':2, 'NegFS':3, 'AltCheckerBoard':4, 'PN23':5, 'PN9':6, 'OneZeroWordToggle':7, 'UserInput':8, 'OneZeroBitToggle':9 })
    top.define_enum('ClockDivideChEnum',{ '{}'.format(i+1):i for i in range(8)})
    top.define_enum('OutputFormatEnum', { 'TwosComplement':0, 'OffsetBinary':1})

    #for i in range(10):
    if False:
        base = 'expert.EpixQuad.Ad9249Readout[{}].'.format(i)
        top.set(base+'FrameDelay', 0, 'UINT16')
        top.set(base+'Invert'    , 0, 'boolEnum')
        for i in range(8):
            top.set(base+f'ChannelDelay[{i}]',0,'UINT16')

        base = 'expert.EpixQuad.Ad9249Config[{}].'.format(i)
        #top.set(base+'ChipId:RO', '', 'CHARSTR')
        top.set(base+'ExternalPdwnMode', 0, 'ExtPwdnEnum')
        top.set(base+'InternalPdwnMode', 0, 'IntPwdnEnum')
        top.set(base+'DutyCycleStabilizer', 0, 'OffOnEnum')
        top.set(base+'ClockDivide', 0, 'ClockDivideChEnum')
        top.set(base+'ChopMode', 0, 'OffOnEnum')
        #top.set(base+'DevIndexMask_DataCh', [0x0,0x0], 'UINT8')
        #top.set(base+'DevIndexMask_FCO', 0x0, 'UINT8')
        #top.set(base+'DevIndexMask_DCO', 0x0, 'UINT8')
        top.set(base+'UserTestModeCfg', 0, 'UserTestModeEnum')
        top.set(base+'OutputTestMode', 0, 'OutputTestModeEnum')
        top.set(base+'OffsetAdjust', 0, 'UINT8')
        top.set(base+'OutputInvert', 0, 'boolEnum')
        top.set(base+'OutputFormat', 1, 'OutputFormatEnum')
        top.set(base+'UserPatt1Lsb', 0, 'UINT8')
        top.set(base+'UserPatt1Msb', 0, 'UINT8')
        top.set(base+'UserPatt2Lsb', 0, 'UINT8')
        top.set(base+'UserPatt2Msb', 0, 'UINT8')
        top.set(base+'LvdsLsbFirst', 0, 'boolEnum')

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
    parser.add_argument('--prod', help='production', action='store_true')
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

    db = 'configdb' if args.prod else 'devconfigdb'
    mycdb = cdb.configdb(f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/', args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config('epixquad')

    top = epixquad_cdict()
    top.setInfo('epixquad', args.name, args.segm, args.id, 'No comment')
    mycdb.modify_device(args.alias, top)
