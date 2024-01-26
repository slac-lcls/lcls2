from psdaq.configdb.typed_json import cdict
from psdaq.configdb.tsdef import *
import psdaq.configdb.configdb as cdb
import pyrogue as pr
import numpy as np
import functools
import sys
import IPython
import argparse

numAsics = 4
elemRows = 384
elemCols = 192

def epixm320_cdict(prjCfg):

    top = cdict()
    top.setAlg('config', [0,0,0])

    top.set("firmwareBuild:RO"  , "-", 'CHARSTR')
    top.set("firmwareVersion:RO",   0, 'UINT32')

    help_str  = "-- user interface --"
    help_str += "\nstart_ns     : nanoseconds to exposure start"
    top.set("help:RO", help_str, 'CHARSTR')

    top.set("user.start_ns", 107749, 'UINT32')

    top.set("user.asic_enable", (1<<numAsics)-1, 'UINT32')

    # timing system
    # run trigger
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].PauseThreshold',16,'UINT32')
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].TriggerDelay',42,'UINT32')
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].Partition',0,'UINT32')
    # daq trigger
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].PauseThreshold',16,'UINT32')
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].TriggerDelay',42,'UINT32')
    top.set('expert.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].Partition',0,'UINT32')

    top.define_enum('rateEnum', fixedRateHzToMarker)
    top.set('expert.App.TimingRx.XpmMiniWrapper.XpmMini.Config_L0Select_RateSel',0,'rateEnum')

    top.define_enum('boolEnum', {'False':0, 'True':1})

    #  There might be Json2Xtc problems with boolEnum in this object
    for i in range(numAsics):
        base = 'expert.App.Mv2Asic[{}].'.format(i)
        top.set(base+'shvc_DAC',            0, 'UINT8')
        top.set(base+'fastPP_enable',       0, 'boolEnum')
        top.set(base+'PulserSync',          0, 'boolEnum')
        top.set(base+'Pll_RO_Reset',        0, 'boolEnum')
        top.set(base+'Pll_Itune',           0, 'UINT8')
        top.set(base+'Pll_KVCO',            0, 'UINT8')
        top.set(base+'Pll_filter1LSB',      0, 'UINT8')
        top.set(base+'Pll_filter1MSB',      0, 'UINT8')
        top.set(base+'Pulser',              0, 'UINT8')
        top.set(base+'pbit',                0, 'boolEnum')
        top.set(base+'atest',               0, 'boolEnum')
        top.set(base+'test',                0, 'boolEnum')
        top.set(base+'sab_test',            0, 'boolEnum')
        top.set(base+'hrtest',              0, 'boolEnum')
        top.set(base+'PulserR',             0, 'boolEnum')
        top.set(base+'DigMon1',             0, 'UINT8')
        top.set(base+'DigMon2',             0, 'UINT8')
        top.set(base+'PulserDac',           0, 'UINT8')
        top.set(base+'MonostPulser',        0, 'UINT8')
        top.set(base+'RefGenB',             0, 'UINT8')
        top.set(base+'Dm1En',               0, 'boolEnum')
        top.set(base+'Dm2En',               0, 'boolEnum')
        top.set(base+'emph_bd',             0, 'UINT8')
        top.set(base+'emph_bc',             0, 'UINT8')
        top.set(base+'VRef_DAC',            0, 'UINT8')
        top.set(base+'VRefLow',             0, 'UINT8')
        top.set(base+'trbit',               0, 'boolEnum')
        top.set(base+'TpsMux',              0, 'UINT8')
        top.set(base+'RoMonost',            0, 'UINT8')
        top.set(base+'TpsGr',               0, 'UINT8')
        top.set(base+'Balcony_clk',         0, 'UINT8')
        top.set(base+'PpOcbS2d',            0, 'boolEnum')
        top.set(base+'Ocb',                 0, 'UINT8')
        top.set(base+'Monost',              0, 'UINT8')
        top.set(base+'mTest',               0, 'boolEnum')
        top.set(base+'Preamp',              0, 'UINT8')
        top.set(base+'S2D_1_b',             0, 'UINT8')
        top.set(base+'Vld1_b',              0, 'UINT8')
        top.set(base+'CompTH_DAC',          0, 'UINT8')
        top.set(base+'loop_mode_sel',       0, 'UINT8')
        top.set(base+'TC',                  0, 'UINT8')
        top.set(base+'S2d',                 0, 'UINT8')
        top.set(base+'S2dDacBias',          0, 'UINT8')
        top.set(base+'Tsd_Tser',            0, 'UINT8')
        top.set(base+'Tps_DAC',             0, 'UINT8')
        top.set(base+'PLL_RO_filter2',      0, 'UINT8')
        top.set(base+'PLL_RO_divider',      0, 'UINT8')
        top.set(base+'TestBe',              0, 'boolEnum')
        top.set(base+'DigRO_disable',       0, 'boolEnum')
        top.set(base+'DelExec',             0, 'boolEnum')
        top.set(base+'DelCCKReg',           0, 'boolEnum')
        top.set(base+'RO_rst_en',           0, 'boolEnum')
        top.set(base+'SlvdsBit',            0, 'boolEnum')
        top.set(base+'FE_Autogain',         0, 'boolEnum')
        top.set(base+'FE_Lowgain',          0, 'boolEnum')
        top.set(base+'RowStartAddr',        0, 'UINT8')
        top.set(base+'RowStopAddr',         0, 'UINT8')
        top.set(base+'ColStartAddr',        0, 'UINT8')
        top.set(base+'ColStopAddr',         0, 'UINT8')
        top.set(base+'DCycle_DAC',          0, 'UINT8')
        top.set(base+'DCycle_en',           0, 'boolEnum')
        top.set(base+'DCycle_bypass',       0, 'boolEnum')
        top.set(base+'Debug_bit',           0, 'UINT8')
        top.set(base+'OSRsel',              0, 'boolEnum')
        top.set(base+'SecondOrder',         0, 'boolEnum')
        top.set(base+'DHg',                 0, 'boolEnum')
        top.set(base+'RefGenC',             0, 'UINT8')
        top.set(base+'dbus_del_sel',        0, 'boolEnum')
        top.set(base+'SDclk_b',             0, 'UINT8')
        top.set(base+'SDrst_b',             0, 'UINT8')
        top.set(base+'Filter_DAC',          0, 'UINT8')
        top.set(base+'Ref_gen_d',           0, 'UINT8')
        top.set(base+'CompEn',              0, 'UINT8')
        top.set(base+'Pixel_CB',            0, 'UINT8')
        top.set(base+'InjEn_ePixM',         0, 'boolEnum')
        top.set(base+'ClkInj_ePixM',        0, 'boolEnum')
        top.set(base+'rowCK2matrix_delay',  0, 'UINT8')
        top.set(base+'DigRO_disable_4b',    0, 'UINT8')
        top.set(base+'RefinN',              0, 'UINT8')
        top.set(base+'RefinCompN',          0, 'UINT8')
        top.set(base+'RefinP',              0, 'UINT8')
        top.set(base+'ro_mode_i',           0, 'UINT8')
        top.set(base+'SDclk1_b',            0, 'UINT8')
        top.set(base+'SDrst1_b',            0, 'UINT8')
        top.set(base+'SDclk2_b',            0, 'UINT8')
        top.set(base+'SDrst2_b',            0, 'UINT8')
        top.set(base+'SDclk3_b',            0, 'UINT8')
        top.set(base+'SDrst3_b',            0, 'UINT8')
        top.set(base+'CompTH_ePixM',        0, 'UINT8')
        top.set(base+'Precharge_DAC_ePixM', 0, 'UINT8')
        top.set(base+'FE_CLK_dly',          0, 'UINT8')
        top.set(base+'FE_CLK_cnt_en',       0, 'boolEnum')
        top.set(base+'FE_ACQ2GR_en',        0, 'boolEnum')
        top.set(base+'FE_sync2GR_en',       0, 'boolEnum')
        top.set(base+'FE_ACQ2InjEn',        0, 'boolEnum')
        top.set(base+'pipoclk_delay_row0',  0, 'UINT8')
        top.set(base+'pipoclk_delay_row1',  0, 'UINT8')
        top.set(base+'pipoclk_delay_row2',  0, 'UINT8')
        top.set(base+'pipoclk_delay_row3',  0, 'UINT8')

    for i in range(numAsics):
        base = 'expert.App.SspMonGrp[{}].'.format(i)
        for j in range(24):
            top.set(base+'UsrDlyCfg[{}]'.format(j), 0, 'UINT16')
        top.set(base+'EnUsrDlyCfg'          , 0, 'UINT8')
        top.set(base+'MinEyeWidth'          , 0, 'UINT8')
        top.set(base+'LockingCntCfg'        , 0, 'UINT32')
        top.set(base+'BypFirstBerDet'       , 0, 'UINT8')
        top.set(base+'Polarity'             , 0, 'UINT32')
        top.set(base+'GearboxSlaveBitOrder' , 0, 'UINT8')
        top.set(base+'GearboxMasterBitOrder', 0, 'UINT8')
        top.set(base+'MaskOffCodeErr'       , 0, 'UINT8')
        top.set(base+'MaskOffDispErr'       , 0, 'UINT8')
        top.set(base+'MaskOffOutOfSync'     , 0, 'UINT8')
        top.set(base+'LockOnIdleOnly'       , 0, 'UINT8')
        top.set(base+'RollOverEn'           , 0, 'UINT8')

    base = 'expert.App.AsicTop.RegisterControlDualClock.'
    top.define_enum('debugChEnum', { 'AsicDM(0)'        :  0,
                                     'AsicDM(1)'        :  1,
                                     'AsicSync'         :  2,
                                     'AsicAcq'          :  3,
                                     'AsicSR0'          :  4,
                                     'AsicGRst'         :  5,
                                     'AsicClkEn'        :  6,
                                     'AsicR0'           :  7,
                                     'AsicSaciCmd(0)'   :  8,
                                     'AsicSaciClk'      :  9,
                                     'AsicSaciSelL(0)'  : 10,
                                     'AsicSaciSelL(1)'  : 11,
                                     'AsicSaciSelL(2)'  : 12,
                                     'AsicSaciSelL(3)'  : 13,
                                     'AsicRsp'          : 14,
                                     'LdoShutDnl0'      : 15,
                                     'LdoShutDnl1'      : 16,
                                     'pllLolL'          : 17,
                                     'biasDacDin'       : 18,
                                     'biasDacSclk'      : 19,
                                     'biasDacCsb'       : 20,
                                     'biasDacClrb'      : 21,
                                     'hsDacCsb'         : 22,
                                     'hsDacSclk'        : 23,
                                     'hsDacDin'         : 24,
                                     'hsLdacb'          : 25,
                                     'slowAdcDout(0)'   : 26,
                                     'slowAdcDrdyL(0)'  : 27,
                                     'slowAdcSyncL(0)'  : 28,
                                     'slowAdcSclk(0)'   : 29,
                                     'slowAdcCsL(0)'    : 30,
                                     'slowAdcDin(0)'    : 31,
                                     'slowAdcRefClk(0)' : 32,
                                     'slowAdcDout(1)'   : 33,
                                     'slowAdcDrdyL(1)'  : 34,
                                     'slowAdcSyncL(1)'  : 35,
                                     'slowAdcSclk(1)'   : 36,
                                     'slowAdcCsL(1)'    : 37,
                                     'slowAdcDin(1)'    : 38,
                                     'slowAdcRefClk(1)' : 39})
    top.set(base+'IDreset',           0, 'UINT32')
    top.set(base+'GlblRstPolarityN',  0, 'boolEnum')
    top.set(base+'ClkSyncEn',         0, 'boolEnum')
    top.set(base+'RoLogicRstN',       0, 'boolEnum')
    top.set(base+'SyncPolarity',      0, 'boolEnum')
    top.set(base+'SyncDelay',         0, 'UINT32')
    top.set(base+'SyncWidth',         0, 'UINT32')
    top.set(base+'SR0Polarity',       0, 'boolEnum')
    top.set(base+'SR0Delay1',         0, 'UINT32')
    top.set(base+'SR0Width1',         0, 'UINT32')
    top.set(base+'ePixAdcSHPeriod',   0, 'UINT16')
    top.set(base+'ePixAdcSHOffset',   0, 'UINT16')
    top.set(base+'AcqPolarity',       0, 'boolEnum')
    top.set(base+'AcqDelay1',         0, 'UINT32')
    top.set(base+'AcqWidth1',         0, 'UINT32')
    top.set(base+'AcqDelay2',         0, 'UINT32')
    top.set(base+'AcqWidth2',         0, 'UINT32')
    top.set(base+'R0Polarity',        0, 'boolEnum')
    top.set(base+'R0Delay',           0, 'UINT32')
    top.set(base+'R0Width',           0, 'UINT32')
    top.set(base+'PPbePolarity',      0, 'boolEnum')
    top.set(base+'PPbeDelay',         0, 'UINT32')
    top.set(base+'PPbeWidth',         0, 'UINT32')
    top.set(base+'PpmatPolarity',     0, 'boolEnum')
    top.set(base+'PpmatDelay',        0, 'UINT32')
    top.set(base+'PpmatWidth',        0, 'UINT32')
    top.set(base+'SaciSyncPolarity',  0, 'boolEnum')
    top.set(base+'SaciSyncDelay',     0, 'UINT32')
    top.set(base+'SaciSyncWidth',     0, 'UINT32')
    top.set(base+'ResetCounters',     0, 'boolEnum')
    top.set(base+'AsicPwrEnable',     0, 'boolEnum')
    top.set(base+'AsicPwrManual',     0, 'boolEnum')
    top.set(base+'AsicPwrManualDig',  0, 'boolEnum')
    top.set(base+'AsicPwrManualAna',  0, 'boolEnum')
    top.set(base+'AsicPwrManualIo',   0, 'boolEnum')
    top.set(base+'AsicPwrManualFpga', 0, 'boolEnum')
    top.set(base+'DebugSel0',         0, 'debugChEnum')
    top.set(base+'DebugSel1',         0, 'debugChEnum')
    top.set(base+'getSerialNumbers',  0, 'boolEnum')
    top.set(base+'AsicRdClk',         0, 'boolEnum')

    base = 'expert.App.AsicTop.TriggerRegisters.'
    top.set(base+'RunTriggerEnable',       0, 'boolEnum')
    top.set(base+'TimingRunTriggerEnable', 0, 'boolEnum')
    top.set(base+'RunTriggerDelay',        0, 'UINT32')
    top.set(base+'DaqTriggerEnable',       0, 'boolEnum')
    top.set(base+'TimingDaqTriggerEnable', 0, 'boolEnum')
    top.set(base+'DaqTriggerDelay',        0, 'UINT32')
    top.set(base+'AutoRunEn',              0, 'boolEnum')
    top.set(base+'AutoDaqEn',              0, 'boolEnum')
    top.set(base+'AutoTrigPeriod',         156250, 'UINT32')
    top.set(base+'PgpTrigEn',              1, 'boolEnum')
    top.set(base+'numberTrigger',          0, 'UINT32')

    for i in range(numAsics):
        base = 'expert.App.AsicTop.DigAsicStrmRegisters{}.'.format(i)
        top.set(base+'asicDataReq'     , 0, 'UINT16')
        top.set(base+'DisableLane'     , 0, 'UINT32')
        top.set(base+'EnumerateDisLane', 0, 'UINT32')

    # These registers are set in the epixm320_config.py file
    #for i in range(numAsics):
    #    base = 'expert.App.AsicTop.BatcherEventBuilder{}.'.format(i)
    #    top.set(base+'Bypass'     , 0, 'UINT8')
    #    top.set(base+'Timeout'    , 0, 'UINT32')
    #    top.set(base+'Blowoff'    , 0, 'boolEnum')

    base = 'expert.App.PowerControl.'
    top.set(base+'DigitalSupplyEn', 3, 'UINT8')

    base = 'expert.App.Dac.Max5443.'
    top.set(base+'Dac[0]', 0, 'UINT16')

    top.define_enum('wfEnum', {'CustomWF':0, 'RampCounter':1})

    base = 'expert.App.Dac.FastDac.'
    top.set(base+'WFEnabled'       , 0,       'boolEnum')
    top.set(base+'run'             , 0,       'boolEnum')
    top.set(base+'externalUpdateEn', 0,       'boolEnum')
    top.set(base+'waveformSource'  , 0,       'wfEnum')
    top.set(base+'samplingCounter' , 0x220,   'UINT16')
    top.set(base+'DacValue'        , 0x30000, 'UINT32')
    top.set(base+'rCStartValue'    , 0,       'UINT32')
    top.set(base+'rCStopValue'     , 0,       'UINT32')
    top.set(base+'rCStep'          , 0,       'UINT32')

    top.define_enum('clkEnum', {'250MHz':1, '125MHz':2, '168MHz':3, 'Default':4})
    base = 'expert.Pll.'
    top.set(base+'Clock', 4, 'clkEnum')
    conv = functools.partial(int, base=16)
    top.set(base+'_250_MHz', np.loadtxt(prjCfg+'/EPixHRM320KPllConfig250Mhz.csv', dtype='uint16', delimiter=',', skiprows=1, converters=conv))
    top.set(base+'_125_MHz', np.loadtxt(prjCfg+'/EPixHRM320KPllConfig125Mhz.csv', dtype='uint16', delimiter=',', skiprows=1, converters=conv))
    top.set(base+'_168_MHz', np.loadtxt(prjCfg+'/EPixHRM320KPllConfig168Mhz.csv', dtype='uint16', delimiter=',', skiprows=1, converters=conv))

    return top


if __name__ == "__main__":
    create = True
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

    args = cdb.createArgs().args

    db = 'configdb' if args.prod else 'devconfigdb'
    url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'
    mycdb = cdb.configdb(url, args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config('epixm320hw')

    if args.dir is None:
        raise Exception('Rogue project root directory is required (--dir)')

    top = epixm320_cdict(args.dir+'/software/config')
    top.setInfo('epixm320hw', args.name, args.segm, args.id, 'No comment')

    mycdb.modify_device(args.alias, top)
