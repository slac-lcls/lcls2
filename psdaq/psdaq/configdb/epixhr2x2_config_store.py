from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import pyrogue as pr
import numpy as np
import sys
import IPython
import argparse

elemRows = 144
elemCols = 192

def epixhr2x2_cdict():

    top = cdict()
    top.setAlg('config', [2,0,0])

    #top.set("firmwareBuild:RO"  , "-", 'CHARSTR')
    #top.set("firmwareVersion:RO",   0, 'UINT32')

    help_str  = "-- user interface --"
    help_str += "\nstart_ns     : nanoseconds to exposure start"
    help_str += "\ngain_mode    : High/Med/Low/AutoHiLo/AutoMedLo/Map"
    help_str += "\npixel_map    : 3D-map of pixel gain/inj settings"
    #top.set("help.user:RO", help_str, 'CHARSTR')

    pixelMap = np.zeros((elemRows*2,elemCols*2),dtype=np.uint8)
    top.set("user.pixel_map", pixelMap)

    top.set("user.start_ns" , 107749, 'UINT32')

    top.define_enum('gainEnum', {'High':0, 'Medium':1, 'Low':2, 'AutoHiLo':3, 'AutoMedLo':4, 'Map':5})
    top.set("user.gain_mode",      0, 'gainEnum')  

    top.set("user.asic_enable"  , 0xf, 'UINT32')

    # timing system
    # run trigger
    top.set('expert.EpixHR.TriggerEventManager.TriggerEventBuffer0.PauseThreshold',16,'UINT32')
    top.set('expert.EpixHR.TriggerEventManager.TriggerEventBuffer0.TriggerDelay',42,'UINT32')
    top.set('expert.EpixHR.TriggerEventManager.TriggerEventBuffer0.Partition',0,'UINT32')
    # daq trigger
    top.set('expert.EpixHR.TriggerEventManager.TriggerEventBuffer1.PauseThreshold',16,'UINT32')
    top.set('expert.EpixHR.TriggerEventManager.TriggerEventBuffer1.TriggerDelay',42,'UINT32')
    top.set('expert.EpixHR.TriggerEventManager.TriggerEventBuffer1.Partition',0,'UINT32')

    top.define_enum('rateEnum', {'929kHz':0, '71kHz':1, '10kHz':2, '1kHz':3, '100Hz':4, '10Hz':5, '1Hz':6})
    top.set('expert.EpixHR.XpmMiniWrapper.XpmMini.Config_L0Select_RateSel',6,'rateEnum')

    top.define_enum('boolEnum', {'False':0, 'True':1})

    base = 'expert.EpixHR.MMCMRegisters.'
    for i in range(7):
        path = base+'CLKOUT{}'.format(i)
        top.set(path+'PhaseMux' , 0, 'UINT8')
        top.set(path+'HighTime' , 2, 'UINT8')
        top.set(path+'LowTime'  , 2, 'UINT8')
        if i==0:
            top.set(path+'Frac'     , 2, 'UINT8')
            top.set(path+'FracEn'   , 1, 'UINT8')
        top.set(path+'Edge'     , 0, 'UINT8')
        top.set(path+'NoCount'  , 0, 'UINT8')
        top.set(path+'DelayTime', 0, 'UINT8')

    base = 'expert.EpixHR.TriggerRegisters.'
    top.set(base+'AutoDaqEn'        , 0, 'boolEnum')
    top.set(base+'AutoRunEn'        , 0, 'boolEnum')
    top.set(base+'AutoTrigPeriod'   , 270000, 'UINT32')
    top.set(base+'PgpTrigEn'        , 1, 'boolEnum')
    top.set(base+'RunTriggerEnable' , 0, 'boolEnum')
    top.set(base+'RunTriggerDelay'  , 0, 'UINT32')
    top.set(base+'DaqTriggerEnable' , 0, 'boolEnum')
    top.set(base+'DaqTriggerDelay'  , 0, 'UINT32')
    
    #  There might be Json2Xtc problems with boolEnum in this object
    for i in range(4):
        base = 'expert.EpixHR.Hr10kTAsic{}.'.format(i)
        top.set(base+'shvc_DAC'      , 0, 'UINT8')
        top.set(base+'fastPP_enable' , 0, 'boolEnum')
        top.set(base+'PulserSync'    , 0, 'boolEnum')
        top.set(base+'Pll_RO_Reset'  , 0, 'boolEnum')
        top.set(base+'Pll_Itune'     , 0, 'UINT8')
        top.set(base+'Pll_KVCO'      , 0, 'UINT8')
        top.set(base+'Pll_filter1LSB', 0, 'UINT8')
        top.set(base+'Pll_filter1MSB', 0, 'UINT8')
        top.set(base+'Pulser'        , 0, 'UINT16')
        top.set(base+'pbit'          , 0, 'boolEnum')
        top.set(base+'atest'         , 0, 'boolEnum')
        top.set(base+'test'          , 0, 'boolEnum')
        top.set(base+'sab_test'      , 0, 'boolEnum')
        top.set(base+'hrtest'        , 0, 'boolEnum')
        top.set(base+'PulserR'       , 0, 'boolEnum')
        top.set(base+'DigMon1'       , 0, 'UINT8')
        top.set(base+'DigMon2'       , 0, 'UINT8')
        top.set(base+'PulserDac'     , 0, 'UINT8')
        top.set(base+'MonostPulser'  , 0, 'UINT8')
        top.set(base+'RefGenB'       , 0, 'UINT8')
        top.set(base+'Dm1En'         , 0, 'boolEnum')
        top.set(base+'Dm2En'         , 0, 'boolEnum')
        top.set(base+'emph_bd'       , 0, 'UINT8')
        top.set(base+'emph_bc'       , 0, 'UINT8')
        top.set(base+'VRef_DAC'      , 0, 'UINT8')
        top.set(base+'VRefLow'       , 0, 'UINT8')
        top.set(base+'trbit'         , 0, 'boolEnum')
        top.set(base+'TpsMux'        , 0, 'UINT8')
        top.set(base+'RoMonost'      , 0, 'UINT8')
        top.set(base+'TpsGr'         , 0, 'UINT8')
        top.set(base+'Balcony_clk'   , 0, 'UINT8')
        top.set(base+'PpOcbS2d'      , 0, 'boolEnum')
        top.set(base+'Ocb'           , 0, 'UINT8')
        top.set(base+'Monost'        , 0, 'UINT8')
        top.set(base+'mTest'         , 0, 'boolEnum')
        top.set(base+'Preamp'        , 0, 'UINT8')
        top.set(base+'S2D_1_b'       , 0, 'UINT8')
        top.set(base+'Vld1_b'        , 0, 'UINT8')
        top.set(base+'CompTH_DAC'    , 0, 'UINT8')
        top.set(base+'TC'            , 0, 'UINT8')
        top.set(base+'S2d'           , 0, 'UINT8')
        top.set(base+'S2dDacBias'    , 0, 'UINT8')
        top.set(base+'Tsd_Tser'      , 0, 'UINT8')
        top.set(base+'Tps_DAC'       , 0, 'UINT8')
        top.set(base+'PLL_RO_filter2', 0, 'UINT8')
        top.set(base+'PLL_RO_divider', 0, 'UINT8')
        top.set(base+'TestBe'        , 0, 'boolEnum')
        top.set(base+'RSTreg'        , 0, 'boolEnum')
        top.set(base+'DelExec'       , 0, 'boolEnum')
        top.set(base+'DelCCKReg'     , 0, 'boolEnum')
        top.set(base+'RO_rst_en'     , 0, 'boolEnum')
        top.set(base+'SlvdsBit'      , 0, 'boolEnum')
        top.set(base+'FELmode'       , 0, 'boolEnum')
        top.set(base+'CompEnOn'      , 0, 'boolEnum')
        top.set(base+'RowStartAddr'  , 0, 'UINT8')
        top.set(base+'RowStopAddr'   , 0, 'UINT8')
        top.set(base+'ColStartAddr'  , 0, 'UINT8')
        top.set(base+'ColStopAddr'   , 0, 'UINT8')
        top.set(base+'DCycle_DAC'    , 0, 'UINT8')
        top.set(base+'DCycle_en'     , 0, 'boolEnum')
        top.set(base+'DCycle_bypass' , 0, 'boolEnum')
        top.set(base+'Debug_bit'     , 0, 'UINT8')
        top.set(base+'OSRsel'        , 0, 'boolEnum')
        top.set(base+'SecondOrder'   , 0, 'boolEnum')
        top.set(base+'DHg'           , 0, 'boolEnum')
        top.set(base+'RefGenC'       , 0, 'UINT8')
        top.set(base+'dbus_del_sel'  , 0, 'boolEnum')
        top.set(base+'SDclk_b'       , 0, 'UINT8')
        top.set(base+'SDrst_b'       , 0, 'UINT8')
        top.set(base+'Filter_DAC'    , 0, 'UINT8')
        top.set(base+'Rodis01'       , 0, 'UINT8')
        top.set(base+'CompEn'        , 0, 'UINT8')
        top.set(base+'Pixel_CB'      , 0, 'UINT8')
        top.set(base+'rodis34'       , 0, 'UINT8')
        top.set(base+'rowCK2matrix_delay' , 0, 'UINT8')
        top.set(base+'ro_mode'       , 0, 'UINT8')
        top.set(base+'rodis5'        , 0, 'UINT8')
        top.set(base+'pipoclk_delay' , 0, 'UINT8')
        top.set(base+'WritePixelData', 0, 'UINT8')

    base = 'expert.EpixHR.RegisterControl.'
    top.define_enum('debugSelEnum', {'Asic01DM':0, 
                                     'AsicSync':1,
                                     'AsicAcq':2,
                                     'AsicSR0':3,
                                     'SaciClk':4,
                                     'SaciCmd':5,
                                     'SaciResp':6,
                                     'SaciSelL0':7,
                                     'SaciSelL1':8,
                                     'AsicRdClk':9,
                                     'deserClk':10,
                                     'WFdacDin':11,
                                     'WFdacSclk':12,
                                     'WFdacCsl':13,
                                     'WFdacLdacL':14,
                                     'WFdacCrtlL':15,
                                     'AsicGRst':16,
                                     'AsicR0':17,
                                     'SlowAdcDin':18,
                                     'SlowAdcDrdy':19,
                                     'SlowAdcDout':20,
                                     'slowAdcRefClk':21,
                                     'pgpTrigger':22,
                                     'acqStart':23})
    top.set(base+'Version'          , 0, 'UINT32')
    top.set(base+'GlblRstPolarity'  , 0, 'boolEnum')
    top.set(base+'ClkSyncEn'        , 0, 'boolEnum')
    top.set(base+'SyncPolarity'     , 0, 'boolEnum')
    top.set(base+'SyncDelay'        , 0, 'UINT32')
    top.set(base+'SyncWidth'        , 0, 'UINT32')
    top.set(base+'SR0Polarity'      , 0, 'boolEnum')
    top.set(base+'SR0Delay1'        , 0, 'UINT32')
    top.set(base+'SR0Width1'        , 0, 'UINT32')
    top.set(base+'ePixAdcSHPeriod'  , 0, 'UINT16')
    top.set(base+'ePixAdcSHOffset'  , 0, 'UINT16')
    top.set(base+'AcqPolarity'      , 0, 'boolEnum')
    top.set(base+'AcqDelay1'        , 0, 'UINT32')
    top.set(base+'AcqWidth1'        , 0, 'UINT32')
    top.set(base+'AcqDelay2'        , 0, 'UINT32')
    top.set(base+'AcqWidth2'        , 0, 'UINT32')
    top.set(base+'R0Polarity'       , 0, 'boolEnum')
    top.set(base+'R0Delay'          , 0, 'UINT32')
    top.set(base+'R0Width'          , 0, 'UINT32')
    top.set(base+'PPbePolarity'     , 0, 'boolEnum')
    top.set(base+'PPbeDelay'        , 0, 'UINT32')
    top.set(base+'PPbeWidth'        , 0, 'UINT32')
    top.set(base+'PpmatPolarity'    , 0, 'boolEnum')
    top.set(base+'PpmatDelay'       , 0, 'UINT32')
    top.set(base+'PpmatWidth'       , 0, 'UINT32')
    top.set(base+'SaciSyncPolarity' , 0, 'boolEnum')
    top.set(base+'SaciSyncDelay'    , 0, 'UINT32')
    top.set(base+'SaciSyncWidth'    , 0, 'UINT32')
    top.set(base+'ResetCounters'    , 0, 'boolEnum')
    top.set(base+'AsicPwrEnable'    , 0, 'boolEnum')
    top.set(base+'AsicPwrManual'    , 0, 'boolEnum')
    top.set(base+'AsicPwrManualDig' , 0, 'boolEnum')
    top.set(base+'AsicPwrManualAna' , 0, 'boolEnum')
    top.set(base+'AsicPwrManualIo'  , 0, 'boolEnum')
    top.set(base+'AsicPwrManualFpga', 0, 'boolEnum')
    top.set(base+'DebugSel_TG'      , 0, 'debugSelEnum')
    top.set(base+'DebugSel_MPS'     , 0, 'debugSelEnum')
    top.set(base+'StartupReq'       , 0, 'boolEnum')

    base = 'expert.EpixHR.PowerSupply.'
    top.set(base+'DigitalEn', 1, 'boolEnum')
    top.set(base+'AnalogEn' , 1, 'boolEnum')

    base = 'expert.EpixHR.HSDac.'
    top.set(base+'WFEnabled'       , 0, 'boolEnum')
    top.set(base+'run'             , 0, 'boolEnum')
    top.set(base+'externalUpdateEn', 0, 'boolEnum')
    top.set(base+'waveformSource'  , 0, 'boolEnum')
    top.set(base+'WFEnabled'       , 0, 'UINT8')
    top.set(base+'samplingCounter' , 0, 'UINT16')
    top.set(base+'DacValue'        , 0, 'UINT16')
    top.set(base+'DacChannel'      , 0, 'UINT8')
    top.set(base+'rCStartValue'    , 0, 'UINT16')
    top.set(base+'rCStopValue'     , 0, 'UINT16')
    top.set(base+'rCStep'          , 0, 'UINT16')

    base = 'expert.EpixHR.SlowDacs.'
    for i in range(5):
        top.set(base+'dac_{}'.format(i), 0, 'UINT16')
    top.set(base+'dummy', 0, 'UINT32')

    base = 'expert.EpixHR.Oscilloscope.'
    top.define_enum('trigPolarityEnum', {'Falling':0, 'Rising':1})
    top.define_enum('trigChannelEnum' , {'TrigReg':0, 
                                         'ThresholdChA':1, 
                                         'ThresholdChB':2,
                                         'AcqStart':3,
                                         'AsicAcq':4,
                                         'AsicR0':5,
                                         'AsicRoClk':6,
                                         'AsicPpmat':7,
                                         'PgpTrigger':8,
                                         'AsicSync':9,
                                         'AsicGR':10,
                                         'AsicSaciSel0':11,
                                         'AsicSaciSel1':12})
    top.define_enum('trigModeEnum', {'Disable':0, 'Axil':1, 'Trig':2 })
    top.define_enum('inputChannelEnum', {'Asic0TpsMux':0,
                                         'Asic1TpsMux':1,
                                         'Asic1TpsMux':2,
                                         'Asic3TpsMux':3})
    top.set(base+'ArmReg'          , 0, 'boolEnum')
    top.set(base+'TrigReg'         , 0, 'boolEnum')
    top.set(base+'ScopeEnable'     , 0, 'boolEnum')
    top.set(base+'TriggerEdge'     , 0, 'trigPolarityEnum')
    top.set(base+'TriggerChannel'  , 0, 'trigChannelEnum')
    top.set(base+'TriggerMode'     , 0, 'trigModeEnum')
    top.set(base+'TriggerAdcThresh', 0, 'UINT16')
    top.set(base+'TriggerHoldoff'  , 0, 'UINT16')
    top.set(base+'TriggerOffset'   , 0, 'UINT16')
    top.set(base+'TraceLength'     , 0, 'UINT16')
    top.set(base+'SkipSamples'     , 0, 'UINT16')
    top.set(base+'InputChannelA'   , 0, 'inputChannelEnum')
    top.set(base+'InputChannelB'   , 0, 'inputChannelEnum')
    top.set(base+'TriggerDelay'    , 0, 'UINT16')

    base = 'expert.EpixHR.FastADCsDebug.'
    for i in range(4):
        top.set(base+'DelayAdc{}_'.format(i)    , 0, 'UINT16')
    top.set(base+'DelayAdcF_', 0, 'UINT16')
    top.set(base+'lockedCountRst', 0, 'boolEnum')
    top.set(base+'FreezeDebug'   , 0, 'boolEnum')

    base = 'expert.EpixHR.Ad9249Config_Adc_0.'
    top.define_enum('ExtPwdnEnum',{'FullPowerDown':0,'Standby':1})
    top.define_enum('IntPwdnEnum',{'ChipRun':0, 'FullPowerDown':1, 'Standby':2, 'DigitalReset':3 })
    top.define_enum('OffOnEnum',{'Off':0, 'On':1})
    top.define_enum('UserTestModeEnum',{ 'single':0, 'alternate':1, 'single_once':2, 'alternate_once':3 })
    top.define_enum('OutputTestModeEnum', { 'Off':0, 'MidscaleShort':1, 'PosFS':2, 'NegFS':3, 'AltCheckerBoard':4, 'PN23':5, 'PN9':6, 'OneZeroWordToggle':7, 'UserInput':8, 'OneZeroBitToggle':9 })
    top.define_enum('ClockDivideChEnum',{ '{}'.format(i+1):i for i in range(8)})
    top.define_enum('OutputFormatEnum', { 'TwosComplement':0, 'OffsetBinary':1})
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

    base = 'expert.EpixHR.SlowAdcRegisters.'
    top.set(base+'StreamEn', 0, 'boolEnum')
    top.set(base+'StreamPeriod', 0, 'UINT32')

    base = 'expert.EpixHR.SspLowSpeedDecoderReg.'
    top.set(base+'EnUsrDlyCfg'          , 0, 'UINT8')
    top.set(base+'UsrDlyCfg'            , 0, 'UINT16')
    top.set(base+'MinEyeWidth'          , 0, 'UINT8')
    top.set(base+'LockingCntCfg'        , 0, 'UINT32')
    top.set(base+'BypFirstBerDet'       , 0, 'UINT8')
    top.set(base+'Polarity'             , 0, 'UINT32')
    top.set(base+'GearBoxSlaveBitOrder' , 0, 'UINT8')
    top.set(base+'GearBoxMasterBitOrder', 0, 'UINT8')
    top.set(base+'MaskOffCodeErr'       , 0, 'UINT8')
    top.set(base+'MaskOffDispErr'       , 0, 'UINT8')
    top.set(base+'MaskOffOutOfSync'     , 0, 'UINT8')
    top.set(base+'LockOnIdleOnly'       , 0, 'UINT8')
    top.set(base+'RollOverEn'           , 0, 'UINT8')

    for i in range(4):
        base = 'expert.EpixHR.PacketRegisters{}.'.format(i)
        top.set(base+'asicDataReq'     , 0, 'UINT16')
        top.set(base+'DisableLane'     , 0, 'UINT8')
        top.set(base+'EnumerateDisLane', 0, 'UINT8')
        top.set(base+'gainBitRemapped' , 0, 'UINT8')

    return top

if __name__ == "__main__":
    create = True
#    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.
    dbname = 'configdb'     #this is the name of the database running on the server.  Only client care about this name.

    args = cdb.createArgs().args

    db = 'configdb' if args.prod else 'devconfigdb'
    mycdb = cdb.configdb(f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/', args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config('epixhr2x2hw')

    top = epixhr2x2_cdict()
    top.setInfo('epixhr2x2hw', args.name, args.segm, args.id, 'No comment')

    mycdb.modify_device(args.alias, top)
