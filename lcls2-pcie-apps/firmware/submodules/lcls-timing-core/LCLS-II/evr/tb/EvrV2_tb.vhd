-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : EvrV2_tb.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2016-01-04
-- Last update: 2016-05-05
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
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
use work.AxiStreamPkg.all;
--use work.SsiPciePkg.all;
use work.TimingPkg.all;
use work.EvrV2Pkg.all;
--use work.PciPkg.all;
use work.SsiPkg.all;
use work.TPGPkg.all;

entity EvrV2_tb is
end EvrV2_tb;

architecture mapping of EvrV2_tb is

  constant TPD_G : time := 1 ns;
  
  signal evrClk              : sl;
  signal evrRst              : sl;
  signal evrBus              : TimingBusType;
  signal exptBus             : ExptBusType := EXPT_BUS_INIT_C;
  signal txPhyClk            : sl;
  signal txPhyRst            : sl;
  -- Trigger and Sync Port
  signal trigOut             : slv(11 downto 0);
  -- Misc.
  signal cardRst             : sl;
  signal ledRedL             : sl;
  signal ledGreenL           : sl;
  signal ledBlueL            : sl;
  
  constant STROBE_INTERVAL_C : integer := 10;

  constant MY_CHANNEL_CONFIG_INIT_C : EvrV2ChannelConfig := (
    enabled => '1', rateSel => "00" & "0000000" & x"0", destSel => "010" & x"0000",
    bsaEnabled => '0', bsaActiveSetup => (others=>'0'), bsaActiveDelay => (others=>'0'), bsaActiveWidth => (others=>'0'),
    dmaEnabled => '1' );

  constant MY_TRIGGER_CONFIG_INIT_C : EvrV2TriggerConfigType := (
    enabled => '1', polarity => '1', delay => x"0007590", width => x"0000080", channel => (others=>'0') );
  
  signal channelConfig    : EvrV2ChannelConfigArray(ReadoutChannels-1 downto 0) := (others=>MY_CHANNEL_CONFIG_INIT_C);
  signal channelConfigS   : EvrV2ChannelConfigArray(ReadoutChannels-1 downto 0) := (others=>MY_CHANNEL_CONFIG_INIT_C);
  signal triggerConfig    : EvrV2TriggerConfigArray(TriggerOutputs-1 downto 0) := (others=>MY_TRIGGER_CONFIG_INIT_C);
  signal triggerConfigS   : EvrV2TriggerConfigArray(TriggerOutputs-1 downto 0) := (others=>MY_TRIGGER_CONFIG_INIT_C);
  
  signal rStrobe        : slv(ReadoutChannels*STROBE_INTERVAL_C downto 0) := (others=>'0');
  signal timingMsg      : TimingMessageType := TIMING_MESSAGE_INIT_C;
  signal eventSel       : slv(ReadoutChannels-1 downto 0) := (others=>'0');
  
  signal dmaControl : EvrV2DmaControlArray(ReadoutChannels+1 downto 0) := (others=>EVRV2_DMA_CONTROL_INIT_C);
  signal dmaData    : EvrV2DmaDataArray(ReadoutChannels+1 downto 0);

  constant SAXIS_MASTER_CONFIG_C : AxiStreamConfigType := ssiAxiStreamConfig(4);
  
  signal dmaMaster : AxiStreamMasterType;
  signal dmaSlave  : AxiStreamSlaveType := AXI_STREAM_SLAVE_INIT_C;

  signal bsaEnabled : slv(ReadoutChannels-1 downto 0);
  signal anyBsaEnabled : sl;
  
  signal dmaFullThr     : Slv24Array (0 downto 0) := (others=>x"000000");
  signal dmaFullThrS    : Slv24Array (0 downto 0) := (others=>x"000000");

begin  -- rtl

  txPhyClk <= evrClk;
  txPhyRst <= evrRst;
  cardRst  <= '0';
  
  process is
  begin
    evrClk <= '1';
    wait for 2.7 ns;
    evrClk <= '0';
    wait for 2.7 ns;
  end process;

  process is
  begin
    evrRst <= '1';
    wait for 20 ns;
    evrRst <= '0';
    wait for 300 us;
    for i in 0 to ReadoutChannels-1 loop
      channelConfig(i).enabled <= '0';
      wait for 3 us;
    end loop;
    wait;
  end process;

  xpm : block
   type RegType is record
      advance    : sl;
      count      : slv( 7 downto 0);
      addrStrobe : slv( 1 downto 0);
      partStrobe : slv(15 downto 0);
      partIndex  : slv( 3 downto 0);
      tbus       : TimingBusType;
      shift      : slv(TIMING_MESSAGE_BITS_C-1 downto 0);
   end record;
   constant REG_INIT_C : RegType := (
     advance    => '0',
     count      => (others=>'0'),
     addrStrobe => "00",
     partStrobe => (others=>'0'),
     partIndex  => (others=>'0'),
     tbus       => TIMING_BUS_INIT_C,
     shift      => (others=>'0') );

   signal recTimingClk : sl;
   signal recTimingRst : sl;

   signal tpgConfig : TPGConfigType := TPG_CONFIG_INIT_C;
   signal xData     : TimingRxType := TIMING_RX_INIT_C;
   signal fiducial  : sl;
   signal streams   : TimingSerialArray(0 downto 0);
   signal streamIds : Slv4Array        (0 downto 0) := ( (others=>TIMING_STREAM_ID) );
   signal advance   : slv              (0 downto 0);

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal data : TimingRxType := TIMING_RX_INIT_C;
  begin

    evrBus       <= r.tbus;
    
    recTimingClk <= evrClk;
    recTimingRst <= evrRst;

   U_TPG : entity work.TPGMini
      port map ( txClk    => recTimingClk,
                 txRst    => recTimingRst,
                 txRdy    => '1',
                 txData   => data.data,
                 txDataK  => data.dataK,
                 statusO  => open,
                 configI  => tpgConfig );
    
    TimingDeserializer_1 : entity work.TimingDeserializer
    generic map ( STREAMS_C => 1 )
    port map ( clk        => recTimingClk,
               rst        => recTimingRst,
               fiducial   => fiducial,
               streams    => streams,
               streamIds  => streamIds,
               advance    => advance,
               data       => data );

   --  This is the serial to parallel part
   comb: process (r, advance, streams, fiducial) is
      variable v : RegType;
   begin
      v := r;
      v.advance     := advance(0);
      v.tbus.strobe := '0';
      if advance(0)='1' then
        v.shift       := streams(0).data & r.shift(r.shift'left downto 16);
      end if;
      
      if (fiducial='1') then
        v.tbus.strobe  := '1';
        v.tbus.valid   := streams(0).ready;
        v.tbus.message := toTimingMessageType(r.shift(TIMING_MESSAGE_BITS_C-1 downto 0));
      end if;

      rin <= v;
   end process comb;
  
   seq: process (recTimingClk) is
   begin
      if rising_edge(recTimingClk) then
         r <= rin;
      end if;
   end process seq;
  end block;
  
  -- Undefined signals
  ledRedL    <= '1';
  ledGreenL  <= '1';
  ledBlueL   <= '1';
  
  U_Dma : entity work.EvrV2Dma
    generic map ( CHANNELS_C    => ReadoutChannels+2,
                  AXIS_CONFIG_C => SAXIS_MASTER_CONFIG_C )
    port map (    clk        => evrClk,
                  dmaCntl    => dmaControl,
                  dmaData    => dmaData,
                  dmaMaster  => dmaMaster,
                  dmaSlave   => dmaSlave );
  
  U_BsaControl : entity work.EvrV2BsaControl
    generic map ( TPD_G      => TPD_G )
    port map (    evrClk     => evrClk,
                  evrRst     => evrRst,
                  enable     => anyBsaEnabled,
                  strobeIn   => evrBus.strobe,
                  dataIn     => evrBus.message,
                  dmaCntl    => dmaControl     (ReadoutChannels),
                  dmaData    => dmaData        (ReadoutChannels) );

  Loop_BsaCh: for i in 0 to ReadoutChannels-1 generate
    U_EventSel   : entity work.EvrV2EventSelect
      generic map ( TPD_G         => TPD_G )
      port map    ( clk           => evrClk,
                    rst           => evrRst,
                    config        => channelConfigS(i),
                    strobeIn      => rStrobe(i*STROBE_INTERVAL_C),
                    dataIn        => timingMsg,
                    exptIn        => exptBus,
                    selectOut     => eventSel(i) );
    U_BsaChannel : entity work.EvrV2BsaChannel
      generic map ( TPD_G         => TPD_G )
      port map    ( evrClk        => evrClk,
                    evrRst        => evrRst,
                    channelConfig => channelConfigS(i),
                    evtSelect     => eventSel(i),
                    strobeIn      => rStrobe(i*STROBE_INTERVAL_C+1),
                    dataIn        => timingMsg,
                    dmaCntl       => dmaControl(i),
                    dmaData       => dmaData(i) );
  end generate;  -- i

  U_EventDma : entity work.EvrV2EventDma
    generic map ( TPD_G      => TPD_G,
                  CHANNELS_C => ReadoutChannels )
    port map (    clk        => evrClk,
                  rst        => evrBus.strobe,
                  strobe     => rStrobe(ReadoutChannels*STROBE_INTERVAL_C),
                  eventSel   => eventSel,
                  eventData  => timingMsg,
                  dmaCntl    => dmaControl(ReadoutChannels+1),
                  dmaData    => dmaData   (ReadoutChannels+1) );
    
  process (evrClk)
  begin  -- process
    if rising_edge(evrClk) then
      rStrobe    <= rStrobe(rStrobe'left-1 downto 0) & evrBus.strobe;
      if evrBus.strobe='1' then
        timingMsg <= evrBus.message;
      end if;
    end if;
  end process;

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
     
  end generate Sync_Trigger;

  Sync_dmaFullThr : entity work.SynchronizerVector
    generic map ( TPD_G   => TPD_G,
                  WIDTH_G => 24 )
    port map (    clk     => evrClk,
                  rst     => evrRst,
                  dataIn  => dmaFullThr (0),
                  dataOut => dmaFullThrS(0) );

end mapping;
