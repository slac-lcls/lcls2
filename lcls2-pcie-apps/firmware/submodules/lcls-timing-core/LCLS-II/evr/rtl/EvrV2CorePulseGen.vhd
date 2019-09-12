-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : EvrV2CorePulseGen.vhd
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2017-04-25
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
-- Till's version of EvrV2Core
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;
use ieee.NUMERIC_STD.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
--use work.SsiPciePkg.all;
use work.TimingPkg.all;
use work.EvrV2Pkg.all;
--use work.PciPkg.all;
use work.SsiPkg.all;

entity EvrV2CorePulseGen is
  generic (
    TPD_G            : time := 1 ns;
    AXIL_BASE_ADDR_G : slv(31 downto 0) := (others => '0') );
  port (
    -- AXI-Lite and IRQ Interface
    axiClk              : in  sl;
    axiRst              : in  sl;
    axilWriteMaster     : in  AxiLiteWriteMasterType;
    axilWriteSlave      : out AxiLiteWriteSlaveType;
    axilReadMaster      : in  AxiLiteReadMasterType;
    axilReadSlave       : out AxiLiteReadSlaveType;
    irqActive           : in  sl;
    irqEnable           : out sl;
    irqReq              : out sl;
    -- DMA
--    dmaRxIbMaster       : out AxiStreamMasterType;
--    dmaRxIbSlave        : in  AxiStreamSlaveType;
--    dmaRxTranFromPci    : in  TranFromPcieType;
--    dmaReady            : out sl;
    -- EVR Ports
    evrClk              : in  sl;
    evrRst              : in  sl;
    evrBus              : in  TimingBusType;
--    exptBus             : in  ExptBusType;
    gtxDebug            : in  slv(7 downto 0);
    -- Trigger and Sync Port
    syncL               : in  sl;
    trigOut             : out slv(11 downto 0);
    evrModeSel          : in  sl;
    delay_ld            : out slv      (11 downto 0);
    delay_wr            : out Slv6Array(11 downto 0);
    delay_rd            : in  Slv6Array(11 downto 0) );
end EvrV2CorePulseGen;

architecture mapping of EvrV2CorePulseGen is

  constant NUM_AXI_MASTERS_C : natural := 2;
  constant CSR_INDEX_C       : natural := 0;
  constant TRG_INDEX_C       : natural := 1;
  constant DMA_INDEX_C       : natural := 2;

  constant AXI_CROSSBAR_MASTERS_CONFIG_C : AxiLiteCrossbarMasterConfigArray(NUM_AXI_MASTERS_C-1 downto 0) := (
    CSR_INDEX_C      => (
      baseAddr      => (AXIL_BASE_ADDR_G + x"00000000"),
      addrBits      => 9,
      connectivity  => X"0001"),
    TRG_INDEX_C => (
      baseAddr      => (AXIL_BASE_ADDR_G + x"00000200"),
      addrBits      => 9,
      connectivity  => X"0001")
--- DMA_INDEX_C => (
---   baseAddr      => x"00080400",
---   addrBits      => 10,
---   connectivity  => X"0001")
  );
  
  signal mAxiWriteMasters : AxiLiteWriteMasterArray(NUM_AXI_MASTERS_C-1 downto 0);
  signal mAxiWriteSlaves  : AxiLiteWriteSlaveArray (NUM_AXI_MASTERS_C-1 downto 0);
  signal mAxiReadMasters  : AxiLiteReadMasterArray (NUM_AXI_MASTERS_C-1 downto 0);
  signal mAxiReadSlaves   : AxiLiteReadSlaveArray  (NUM_AXI_MASTERS_C-1 downto 0);
  
  constant STROBE_INTERVAL_C : integer := 12;
  
  signal bsaControl       : EvrV2BsaControlType;
  signal bsaChannel       : EvrV2BsaChannelArray   (ReadoutChannels-1 downto 0);
  signal channelConfig    : EvrV2ChannelConfigArray(ReadoutChannels-1 downto 0);
  signal channelConfigS   : EvrV2ChannelConfigArray(ReadoutChannels-1 downto 0) := (others=>EVRV2_CHANNEL_CONFIG_INIT_C);
  signal triggerConfig    : EvrV2TriggerConfigArray(TriggerOutputs-1 downto 0);
  signal triggerConfigS   : EvrV2TriggerConfigArray(TriggerOutputs-1 downto 0) := (others=>EVRV2_TRIGGER_CONFIG_INIT_C);
  
  signal gtxDebugS   : slv(7 downto 0);

  signal rStrobe        : slv(ReadoutChannels*STROBE_INTERVAL_C+34 downto 0) := (others=>'0');
  signal timingMsg      : TimingMessageType := TIMING_MESSAGE_INIT_C;
  signal eventMsg       : slv(TIMING_MESSAGE_BITS_NO_BSA_C-1 downto 0) := (others=>'0');
  signal dmaSel         : slv(ReadoutChannels-1 downto 0) := (others=>'0');
  signal eventSel       : slv(ReadoutChannels-1 downto 0) := (others=>'0');
  signal eventCount     : SlVectorArray(ReadoutChannels downto 0,31 downto 0);
  signal rstCount : sl;
  
  signal dmaCtrl    : AxiStreamCtrlType;
  signal dmaData    : EvrV2DmaDataArray(ReadoutChannels+1 downto 0);

  constant SAXIS_MASTER_CONFIG_C : AxiStreamConfigType := ssiAxiStreamConfig(4);
  
  signal dmaMaster : AxiStreamMasterType;
  signal dmaSlave  : AxiStreamSlaveType;

  signal pciClk : sl;
  signal pciRst : sl;

--  signal rxDescToPci   : DescToPcieType;
--  signal rxDescFromPci : DescFromPcieType;

  signal bsaEnabled : slv(ReadoutChannels-1 downto 0);
  signal anyBsaEnabled : sl;
  
  signal irqRequest : sl;

  signal dmaFullThr     : Slv24Array (0 downto 0);
  signal dmaFullThrS    : Slv24Array (0 downto 0);

  signal partitionAddr  : slv(31 downto 0) := (others=>'1');
  signal modeSel        : sl;
  signal delay_wrb      : Slv6Array(11 downto 0) := (others=>(others=>'0'));
  signal delay_ldb      : slv      (11 downto 0) := (others=>'1');
  
begin  -- rtl

  assert (rStrobe'length <= 200)
    report "rStrobe'length exceeds clocks per cycle"
    severity failure;
  
  pciClk <= axiClk;
  pciRst <= axiRst;
  irqReq <= irqRequest;

  modeSel    <= evrModeSel;
  delay_ld   <= delay_ldb;
  delay_wr   <= delay_wrb;

--  dmaReady <= not dmaCtrl.pause;

  -------------------------
  -- AXI-Lite Crossbar Core
  -------------------------  
  AxiLiteCrossbar_Inst : entity work.AxiLiteCrossbar
    generic map (
      TPD_G              => TPD_G,
      NUM_SLAVE_SLOTS_G  => 1,
      NUM_MASTER_SLOTS_G => NUM_AXI_MASTERS_C,
      MASTERS_CONFIG_G   => AXI_CROSSBAR_MASTERS_CONFIG_C)
    port map (
      axiClk              => axiClk,
      axiClkRst           => axiRst,
      sAxiWriteMasters(0) => axilWriteMaster,
      sAxiWriteSlaves (0) => axilWriteSlave,
      sAxiReadMasters (0) => axilReadMaster,
      sAxiReadSlaves  (0) => axilReadSlave,
      mAxiWriteMasters    => mAxiWriteMasters,
      mAxiWriteSlaves     => mAxiWriteSlaves,
      mAxiReadMasters     => mAxiReadMasters,
      mAxiReadSlaves      => mAxiReadSlaves);   
  
--  U_PciRxDesc : entity work.EvrV2PcieRxDesc
--    generic map ( DMA_SIZE_G       => 1 )
--    port map (    dmaDescToPci  (0)=> rxDescToPci,
--                  dmaDescFromPci(0)=> rxDescFromPci,
--                  axiReadMaster    => mAxiReadMasters (DMA_INDEX_C),
--                  axiReadSlave     => mAxiReadSlaves  (DMA_INDEX_C),
--                  axiWriteMaster   => mAxiWriteMasters(DMA_INDEX_C),
--                  axiWriteSlave    => mAxiWriteSlaves (DMA_INDEX_C),
--                  irqReq           => irqRequest,
--                  cntRst           => '0',
--                  pciClk           => pciClk,
--                  pciRst           => pciRst );
--
--  U_PciRxDma : entity work.EvrV2PcieRxDma
--    generic map ( TPD_G                 => TPD_G,
--                  SAXIS_MASTER_CONFIG_G => SAXIS_MASTER_CONFIG_C,
--                  FIFO_ADDR_WIDTH_G     => 10 )
--    port map (    sAxisClk       => evrClk,
--                  sAxisRst       => evrRst,
--                  sAxisMaster    => dmaMaster,
--                  sAxisSlave     => dmaSlave,
--                  sAxisCtrl      => dmaCtrl,
--                  sAxisPauseThr  => dmaFullThrS(0)(9 downto 0),
--                  pciClk         => pciClk,
--                  pciRst         => pciRst,
--                  dmaIbMaster    => dmaRxIbMaster,
--                  dmaIbSlave     => dmaRxIbSlave,
--                  dmaDescFromPci => rxDescFromPci,
--                  dmaDescToPci   => rxDescToPci,
--                  dmaTranFromPci => dmaRxTranFromPci,
--                  dmaChannel     => x"0" );
--
--  U_Dma : entity work.EvrV2Dma
--    generic map ( CHANNELS_C    => ReadoutChannels+2,
--                  AXIS_CONFIG_C => SAXIS_MASTER_CONFIG_C )
--    port map (    clk        => evrClk,
--                  strobe     => rStrobe(rStrobe'left),
--                  modeSel    => modeSel,
--                  dmaCntl    => dmaCtrl,
--                  dmaData    => dmaData,
--                  dmaMaster  => dmaMaster,
--                  dmaSlave   => dmaSlave );
--  
--  U_BsaControl : entity work.EvrV2BsaControl
--    generic map ( TPD_G      => TPD_G )
--    port map (    evrClk     => evrClk,
--                  evrRst     => evrRst,
--                  enable     => anyBsaEnabled,
--                  strobeIn   => evrBus.strobe,
--                  dataIn     => evrBus.message,
--                  dmaData    => dmaData        (ReadoutChannels) );
--
  Loop_BsaCh: for i in 0 to ReadoutChannels-1 generate
    U_EventSel : entity work.EvrV2EventSelect
      generic map ( TPD_G         => TPD_G )
      port map    ( clk           => evrClk,
                    rst           => evrRst,
                    config        => channelConfigS(i),
                    strobeIn      => rStrobe(i*STROBE_INTERVAL_C+4),
                    dataIn        => timingMsg,
--                    exptIn        => exptBus,
                    selectOut     => eventSel(i),
                    dmaOut        => dmaSel(i) );
--    U_BsaChannel : entity work.EvrV2BsaChannel
--      generic map ( TPD_G         => TPD_G,
--                    CHAN_C        => i )
--      port map    ( evrClk        => evrClk,
--                    evrRst        => evrRst,
--                    channelConfig => channelConfigS(i),
--                    evtSelect     => dmaSel(i),
--                    strobeIn      => rStrobe(i*STROBE_INTERVAL_C+5),
--                    dataIn        => timingMsg,
--                    dmaData       => dmaData(i) );
  end generate;  -- i

--  U_EventDma : entity work.EvrV2EventDma
--    generic map ( TPD_G      => TPD_G,
--                  CHANNELS_C => ReadoutChannels )
--    port map (    clk        => evrClk,
--                  rst        => evrBus.strobe,
--                  strobe     => rStrobe(ReadoutChannels*STROBE_INTERVAL_C+5),
--                  eventSel   => dmaSel,
--                  eventData  => eventMsg,
--                  dmaData    => dmaData   (ReadoutChannels+1) );
--    
  process (evrClk)
    variable acrate : integer;
    variable destn  : integer;
  begin  -- process
    if rising_edge(evrClk) then
      rStrobe    <= rStrobe(rStrobe'left-1 downto 0) & evrBus.strobe;
      if evrBus.strobe='1' then
        if modeSel = '0' then  -- map LCLS timing stream to look like LCLS-II frame
          timingMsg.pulseId     <= resize(evrBus.stream.pulseId,64);
          timingMsg.timeStamp   <= evrBus.stream.dbuff.epicsTime(31 downto 0) &
                                   evrBus.stream.dbuff.epicsTime(63 downto 32);
          timingMsg.bsaInit     <= resize(evrBus.stream.dbuff.edefInit,64);
          -- encode minor/major mask on bsaInit
          timingMsg.bsaActive   <= (resize(evrBus.stream.dbuff.edefMinor,64) and
                                    resize(evrBus.stream.dbuff.edefInit,64)) or
                                   (resize(evrBus.stream.dbuff.dmod(147 downto 128),64) and not
                                    resize(evrBus.stream.dbuff.edefInit,64));
          timingMsg.bsaAvgDone  <= (resize(evrBus.stream.dbuff.edefMajor,64) and
                                    resize(evrBus.stream.dbuff.edefInit,64)) or
                                   (resize(evrBus.stream.dbuff.edefAvgDn,64) and not
                                    resize(evrBus.stream.dbuff.edefInit,64));
          timingMsg.bsaDone     <= resize(evrBus.stream.dbuff.edefAvgDn,64);
          timingMsg.fixedRates  <= (others=>'0');
          timingMsg.acRates(0)  <= '1';
          timingMsg.acRates(5 downto 1) <= evrBus.stream.dbuff.dmod(152 downto 148);
          timingMsg.acTimeSlot <= evrBus.stream.dbuff.dmod(127 downto 125);
          for i in 0 to 15 loop
            timingMsg.control(i) <= evrBus.stream.eventCodes(i*16+15 downto i*16);
          end loop;
          timingMsg.control(16 to 17) <= (others=>(others=>'0'));
          -- Simulate beam request word : charge=0, dest={D10DMP,LI25,UND}, beam=POCKCEL
          destn := 2;
          if evrBus.stream.dbuff.dmod(61)='1' then
            destn := 0;
          end if;
          if evrBus.stream.dbuff.dmod(60)='1' then
            destn := 1;
          end if;
          timingMsg.beamRequest <= x"000000" & toSlv(destn,4) & "000" & evrBus.stream.dbuff.dmod(83);
          --
          eventMsg              <= resize(evrBus.stream.eventCodes,288) &
                                   x"00000000" &
                                   evrBus.stream.dbuff.dmod &
                                   evrBus.stream.dbuff.epicsTime(31 downto 0) &
                                   evrBus.stream.dbuff.epicsTime(63 downto 32) &
                                   resize(evrBus.stream.pulseId,64);

                                   
        else
          timingMsg <= evrBus.message;
          eventMsg  <= toSlvNoBsa(evrBus.message);
        end if;
      end if;
    end if;
  end process;

  SyncVector_Gtx : entity work.SynchronizerVector
    generic map (
      TPD_G          => TPD_G,
      WIDTH_G        => 8)
    port map (
      clk                   => axiClk,
      dataIn                => gtxDebug,
      dataOut               => gtxDebugS );

  Sync_EvtCount : entity work.SyncStatusVector
    generic map ( TPD_G   => TPD_G,
                  WIDTH_G => ReadoutChannels+1 )
    port map    ( statusIn(ReadoutChannels) => evrBus.strobe,
                  statusIn(ReadoutChannels-1 downto 0) => eventSel,
                  cntRstIn     => rstCount,
                  rollOverEnIn => (others=>'1'),
                  cntOut       => eventCount,
                  wrClk        => evrClk,
                  wrRst        => '0',
                  rdClk        => axiClk,
                  rdRst        => axiRst );

  Out_Trigger: for i in 0 to TriggerOutputs-1 generate
     U_Trig : entity work.EvrV2Trigger
        generic map ( TPD_G    => TPD_G,
                      CHANNELS_C => ReadoutChannels,
                      --DEBUG_C    => (i<1) )
                      DEBUG_C    => false )
        port map (    clk      => evrClk,
                      rst      => evrRst,
                      config   => triggerConfigS(i),
                      arm      => eventSel,
                      fire     => evrBus.strobe,
                      trigstate=> trigOut(i) );
  end generate Out_Trigger;
  
  U_EvrAxi : entity work.EvrV2Axi
    generic map ( TPD_G      => TPD_G,
                  CHANNELS_C => ReadoutChannels )
    port map (    axiClk              => axiClk,
                  axiRst              => axiRst,
                  axilWriteMaster     => mAxiWriteMasters (CSR_INDEX_C),
                  axilWriteSlave      => mAxiWriteSlaves  (CSR_INDEX_C),
                  axilReadMaster      => mAxiReadMasters  (CSR_INDEX_C),
                  axilReadSlave       => mAxiReadSlaves   (CSR_INDEX_C),
                  -- configuration
                  irqEnable           => irqEnable,
                  channelConfig       => channelConfig,
                  trigSel             => open,
                  dmaFullThr          => dmaFullThr(0),
                  -- status
                  irqReq              => irqRequest,
                  partitionAddr       => partitionAddr,
                  rstCount            => rstCount,
                  eventCount          => eventCount,
                  gtxDebug            => gtxDebugS );

  U_EvrTrigReg : entity work.EvrV2TrigReg
    generic map ( TPD_G      => TPD_G,
                  TRIGGERS_C => TriggerOutputs )
    port map (    axiClk              => axiClk,
                  axiRst              => axiRst,
                  axilWriteMaster     => mAxiWriteMasters (TRG_INDEX_C),
                  axilWriteSlave      => mAxiWriteSlaves  (TRG_INDEX_C),
                  axilReadMaster      => mAxiReadMasters  (TRG_INDEX_C),
                  axilReadSlave       => mAxiReadSlaves   (TRG_INDEX_C),
                  -- configuration
                  triggerConfig       => triggerConfig,
                  -- status
                  delay_rd            => delay_rd(TriggerOutputs-1 downto 0) );

  anyBsaEnabled <= uOr(bsaEnabled);

  -- Synchronize configurations to evrClk
  Sync_Channel: for i in 0 to ReadoutChannels-1 generate
    
    U_SyncRate : entity work.SynchronizerVector
      generic map ( TPD_G   => TPD_G,
                    WIDTH_G => channelConfig (i).rateSel'length)
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => channelConfig (i).rateSel,
                    dataOut => channelConfigS(i).rateSel );
    
    U_SyncDest : entity work.SynchronizerVector
      generic map ( TPD_G   => TPD_G,
                    WIDTH_G => channelConfig (i).destSel'length)
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => channelConfig (i).destSel,
                    dataOut => channelConfigS(i).destSel );
     
    Sync_Enable : entity work.Synchronizer
      generic map ( TPD_G   => TPD_G )
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => channelConfig (i).enabled,
                    dataOut => channelConfigS(i).enabled );

    Sync_dmaEnable : entity work.Synchronizer
      generic map ( TPD_G   => TPD_G )
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => channelConfig (i).dmaEnabled,
                    dataOut => channelConfigS(i).dmaEnabled );

    Sync_bsaEnable : entity work.Synchronizer
      generic map ( TPD_G   => TPD_G )
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => channelConfig (i).bsaEnabled,
                    dataOut => bsaEnabled(i) );

    channelConfigS(i).bsaEnabled <= bsaEnabled(i);
    
    Sync_Setup : entity work.SynchronizerVector
      generic map ( TPD_G   => TPD_G,
                    WIDTH_G => channelConfig (i).bsaActiveSetup'length)
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => channelConfig (i).bsaActiveSetup,
                    dataOut => channelConfigS(i).bsaActiveSetup );
    
    Sync_Delay : entity work.SynchronizerVector
      generic map ( TPD_G   => TPD_G,
                    WIDTH_G => channelConfig (i).bsaActiveDelay'length)
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => channelConfig (i).bsaActiveDelay,
                    dataOut => channelConfigS(i).bsaActiveDelay );
    
    Sync_Width : entity work.SynchronizerVector
      generic map ( TPD_G   => TPD_G,
                    WIDTH_G => channelConfig (i).bsaActiveWidth'length)
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => channelConfig (i).bsaActiveWidth,
                    dataOut => channelConfigS(i).bsaActiveWidth );
  
  end generate Sync_Channel;

  Sync_Trigger: for i in 0 to TriggerOutputs-1 generate
    
    Sync_Enable : entity work.Synchronizer
      generic map ( TPD_G   => TPD_G )
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => triggerConfig (i).enabled,
                    dataOut => triggerConfigS(i).enabled );

    Sync_Polarity : entity work.Synchronizer
      generic map ( TPD_G   => TPD_G )
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => triggerConfig (i).polarity,
                    dataOut => triggerConfigS(i).polarity );

    Sync_Channel : entity work.SynchronizerVector
      generic map ( TPD_G   => TPD_G,
                    WIDTH_G => triggerConfig (i).channel'length)
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => triggerConfig (i).channel,
                    dataOut => triggerConfigS(i).channel );
    
    U_SyncDelay : entity work.SynchronizerVector
      generic map ( TPD_G   => TPD_G,
                    WIDTH_G => triggerConfig (i).delay'length)
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => triggerConfig (i).delay,
                    dataOut => triggerConfigS(i).delay );
    
    U_SyncWidth : entity work.SynchronizerVector
      generic map ( TPD_G   => TPD_G,
                    WIDTH_G => triggerConfig (i).width'length)
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => triggerConfig (i).width,
                    dataOut => triggerConfigS(i).width );

    U_SyncTap : entity work.SynchronizerVector
      generic map ( TPD_G   => TPD_G,
                    WIDTH_G => triggerConfig (i).delayTap'length)
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => triggerConfig (i).delayTap,
                    dataOut => triggerConfigS(i).delayTap );

    U_SyncTapLd : entity work.SynchronizerOneShot
      generic map ( TPD_G   => TPD_G )
      port map (    clk     => evrClk,
                    rst     => evrRst,
                    dataIn  => triggerConfig (i).loadTap,
                    dataOut => triggerConfigS(i).loadTap );

    delay_wrb(i) <= triggerConfig(i).delayTap;
    delay_ldb(i) <= triggerConfig(i).loadTap;

  end generate Sync_Trigger;

  Sync_dmaFullThr : entity work.SynchronizerVector
    generic map ( TPD_G   => TPD_G,
                  WIDTH_G => 24 )
    port map (    clk     => evrClk,
                  rst     => evrRst,
                  dataIn  => dmaFullThr (0),
                  dataOut => dmaFullThrS(0) );

  --Sync_partAddr : entity work.SynchronizerVector
  --  generic map ( TPD_G   => TPD_G,
  --                WIDTH_G => partitionAddr'length )
  --  port map (    clk     => axiClk,
  --                dataIn  => exptBus.message.partitionAddr,
  --                dataOut => partitionAddr );

end mapping;
