-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmSim.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2018-11-02
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: XpmApp's Top Level
-- 
-- Note: Common-to-XpmApp interface defined here (see URL below)
--       https://confluence.slac.stanford.edu/x/rLyMCw
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 DAQ Software'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 DAQ Software', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use STD.textio.all;
use ieee.std_logic_textio.all;

use work.StdRtlPkg.all;
use work.TimingExtnPkg.all;
use work.TimingPkg.all;
use work.TPGPkg.all;
use work.XpmPkg.all;

library unisim;
use unisim.vcomponents.all;

entity XpmSim is
  generic ( USE_TX_REF        : boolean := false;
            ENABLE_DS_LINKS_G : slv(NDSLinks-1 downto 0) := (others=>'0');
            ENABLE_BP_LINKS_G : slv(NBPLinks-1 downto 0) := (others=>'0');
            RATE_DIV_G        : integer := 4;
            RATE_SELECT_G     : integer := 1;
            TRIG_DELAY_G      : integer := 80;
            PIPELINE_DEPTH_G  : integer := 200 );
  port ( txRefClk     : in  sl := '0';
         dsRxClk      : in  slv       (NDSLinks-1 downto 0);
         dsRxRst      : in  slv       (NDSLinks-1 downto 0);
         dsRxData     : in  Slv16Array(NDSLinks-1 downto 0);
         dsRxDataK    : in  Slv2Array (NDSLinks-1 downto 0);
         dsTxClk      : out slv       (NDSLinks-1 downto 0);
         dsTxRst      : out slv       (NDSLinks-1 downto 0);
         dsTxData     : out Slv16Array(NDSLinks-1 downto 0);
         dsTxDataK    : out Slv2Array (NDSLinks-1 downto 0);
         --
         bpTxClk      : out sl;
         bpTxLinkUp   : in  sl;
         bpTxData     : out slv(15 downto 0);
         bpTxDataK    : out slv( 1 downto 0);
         bpRxClk      : in  sl;
         bpRxClkRst   : in  sl;
         bpRxLinkUp   : in  slv       (NBPLinks-1 downto 0);
         bpRxLinkFull : in  Slv16Array(NBPLinks-1 downto 0) );
end XpmSim;

architecture top_level_app of XpmSim is

   constant GEN_CLEAR_G : boolean := false;
   
   -- Reference Clocks and Resets
   signal recTimingClk : sl;
   signal recTimingRst : sl;
   signal regClk       : sl;
   signal regRst       : sl;
   
   signal tpgConfig : TPGConfigType := TPG_CONFIG_INIT_C;
   signal xpmConfig : XpmConfigType := XPM_CONFIG_INIT_C;
   signal xpmStatus : XpmStatusType;
   signal linkStatus: XpmLinkStatusArray(NDSLinks-1 downto 0) := (others=>XPM_LINK_STATUS_INIT_C);
   
   -- Timing Interface (timingClk domain) 
   signal xData     : TimingRxType := TIMING_RX_INIT_C;
   signal timingBus : TimingBusType;
   signal timingBusL: TimingBusType;
   signal exptBus   : ExptBusType;
   
   signal pconfig : XpmPartitionConfigArray(NPartitions-1 downto 0) := (others=>XPM_PARTITION_CONFIG_INIT_C);
   
    function HexChar(v : in slv(3 downto 0)) return character is
      variable result : character := '0';
    begin
      case(v) is
      when x"0" => result := '0';
      when x"1" => result := '1';
      when x"2" => result := '2';
      when x"3" => result := '3';
      when x"4" => result := '4';
      when x"5" => result := '5';
      when x"6" => result := '6';
      when x"7" => result := '7';
      when x"8" => result := '8';
      when x"9" => result := '9';
      when x"A" => result := 'a';
      when x"B" => result := 'b';
      when x"C" => result := 'c';
      when x"D" => result := 'd';
      when x"E" => result := 'e';
      when x"F" => result := 'f';
      when others => null;
    end case;
    return result;
  end function;

  function HexString(v : in slv(31 downto 0)) return string is
    variable result : string(8 downto 1);
  begin
    for i in 0 to 7 loop
      result(i+1) := HexChar(v(4*i+3 downto 4*i));
    end loop;
    return result;
  end function;

begin

  --  Generate clocks and resets
  process is
  begin
    regRst <= '1';
    wait for 10 ns;
    regRst <= '0';
    wait;
  end process;
  
  process is
  begin
    regClk <= '1';
    wait for 4.0 ns;
    regClk <= '0';
    wait for 4.0 ns;
  end process;

  recTimingRst <= regRst;

  NOGEN_REFCLK : if USE_TX_REF=true generate
    recTimingClk <= txRefClk;
  end generate;

  GEN_REFCLK : if USE_TX_REF=false generate
    process is
    begin
      recTimingClk <= '0';
      wait for 2.692 ns;
      recTimingClk <= '1';
      wait for 2.692 ns;
    end process;
  end generate;
  
  dsTxClk <= (others=>recTimingClk);
  dsTxRst <= (others=>recTimingRst);
  bpTxClk <= recTimingClk;
  
  U_TPG : entity work.TPGMini
    port map ( txClk    => recTimingClk,
               txRst    => recTimingRst,
               txRdy    => '1',
               txData   => xData.data,
               txDataK  => xData.dataK,
               statusO  => open,
               configI  => tpgConfig );

  tpgConfig.FixedRateDivisors(RATE_SELECT_G) <= toSlv(RATE_DIV_G,20);
  tpgConfig.pulseIdWrEn                      <= '0';
  
  xpmConfig.partition <= pconfig;
  xpmConfig.dsLink(0).txDelay <= toSlv(200,20);
  xpmConfig.dsLink(1).txDelay <= toSlv(200,20);

  GEN_DS_ENABLE : for i in 0 to NDSLinks-1 generate
    GEN_ENABLE: if ENABLE_DS_LINKS_G(i)='1' generate
      xpmConfig.dsLink(i).enable     <= '1';
      xpmConfig.dsLink(i).partition  <= toSlv( 0, 4);
    end generate;
  end generate;
   
  GEN_BP_ENABLE : for i in 0 to NBPLinks-1 generate
    GEN_ENABLE: if ENABLE_BP_LINKS_G(i)='1' generate
      xpmConfig.bpLink(i).enable     <= '1';
      xpmConfig.dsLink(i).partition  <= toSlv( 0, 4);
    end generate;
  end generate;

  process is
  begin
     for i in 0 to NPartitions-1 loop
       pconfig(i).pipeline.depth <= toSlv((TRIG_DELAY_G+i)*200,20);
     end loop;
       
     pconfig(0).analysis.rst  <= x"f";
     pconfig(0).analysis.tag  <= x"00000000";
     pconfig(0).analysis.push <= x"0";
     wait for 100 ns;
     pconfig(0).analysis.rst  <= x"0";
     wait for 5000 ns;
     wait until regClk='0';
     pconfig(0).analysis.tag  <= x"00000001";
     pconfig(0).analysis.push <= x"1";
     wait until regClk='1';
     wait until regClk='0';
     pconfig(0).analysis.push <= x"0";
     wait until regClk='1';
     wait until regClk='0';
     pconfig(0).analysis.tag  <= x"00000002";
     pconfig(0).analysis.push <= x"1";
     wait until regClk='1';
     wait until regClk='0';
     pconfig(0).analysis.push <= x"0";
     wait until regClk='1';
     wait until regClk='0';
     pconfig(0).analysis.tag  <= x"00000003";
     pconfig(0).analysis.push <= x"1";
     wait until regClk='1';
     wait until regClk='0';
     pconfig(0).analysis.push <= x"0";
     wait until regClk='1';
     wait until regClk='0';

     wait for 10000 ns;
     for i in 0 to NPartitions-1 loop
       pconfig(i).message.hdr     <= MSG_DELAY_PWORD;
       pconfig(i).message.payload <= toSlv(TRIG_DELAY_G+i,8);
       pconfig(i).message.insert  <= '1';
     end loop;
   
     wait until regClk='1';
     wait until regClk='0';
     
     for i in 0 to NPartitions-1 loop
       pconfig(i).message.insert  <= '0';
     end loop;

     for i in 0 to TRIG_DELAY_G loop
       wait for 1 us;
     end loop;
     
     pconfig(0).l0Select.enabled <= '1';
     pconfig(0).l0Select.rateSel <= toSlv(RATE_SELECT_G,16);
     pconfig(0).l0Select.destSel <= x"8000";
     pconfig(0).inhibit.setup(0).enable   <= '1';
     pconfig(0).inhibit.setup(0).limit    <= toSlv(3,4);
     pconfig(0).inhibit.setup(0).interval <= toSlv(10,12);

     for i in 1 to NPartitions-1 loop
       pconfig(i).l0Select.enabled <= '1';
       pconfig(i).l0Select.rateSel <= toSlv(1,16);
       pconfig(i).l0Select.destSel <= x"8000";
     end loop;

     wait for 20 us;

     pconfig(0).l0Select.enabled <= '0';

     wait for 10 us;

     if GEN_CLEAR_G then
       for i in 0 to NPartitions-1 loop
         pconfig(i).message.hdr     <= MSG_CLEAR_FIFO;
         pconfig(i).message.insert  <= '1';
       end loop;
       
       wait until regClk='1';
       wait until regClk='0';
       
       for i in 0 to NPartitions-1 loop
         pconfig(i).message.insert  <= '0';
       end loop;

       wait for 100 ns;
     end if;
     
     pconfig(0).l0Select.enabled <= '1';
     
     wait;
   end process;

   process
    file     trigs : text;
    variable oline : line;
    variable enabld: sl := '0';
    constant ENABLED_S  : string(7 downto 1) := "enabled";
    constant DISABLED_S : string(8 downto 1) := "disabled";
   begin
     file_open(trigs, "trigs.txt", write_mode);
     loop
       wait until rising_edge(recTimingClk);
       if enabld='0' and pconfig(0).l0Select.enabled='1' then
         write(oline, ENABLED_S);
         writeline(trigs, oline);
       elsif enabld='1' and pconfig(0).l0Select.enabled='0' then
         write(oline, DISABLED_S);
         writeline(trigs, oline);
       end if;
       enabld := pconfig(0).l0Select.enabled;
       
       if pconfig(0).l0Select.enabled='1' and timingBus.strobe='1' and timingBus.message.fixedRates(1)='1' then
         write(oline, HexString(timingBus.message.pulseId(31 downto  0)), right, 9);
         write(oline, HexString(timingBus.message.pulseId(63 downto 32)), right, 9);
         writeline(trigs, oline);
       end if;
     end loop;
     file_close(trigs);
   end process;

   timingBus.stream <= TIMING_STREAM_INIT_C;
   timingBus.v1     <= LCLS_V1_TIMING_DATA_INIT_C;
   timingBus.v2     <= LCLS_V2_TIMING_DATA_INIT_C;

   U_RxLcls2 : entity work.TimingFrameRx
     port map ( rxClk               => recTimingClk,
                rxRst               => recTimingRst,
                rxData              => xData,
                messageDelay        => (others=>'0'),
                messageDelayRst     => '0',
                timingMessage       => timingBus.message,
                timingMessageStrobe => timingBus.strobe,
                timingMessageValid  => timingBus.valid,
                timingExtn          => exptBus.message,
                timingExtnValid     => exptBus.valid );

   U_Application : entity work.XpmApp
      generic map ( NDsLinks => linkStatus'length,
                    NBpLinks => bpRxLinkUp'length )
      port map (
         -----------------------
         -- Application Ports --
         -----------------------
         -- -- AMC's DS Ports
         dsLinkStatus    => linkStatus,
         dsRxData        => dsRxData,
         dsRxDataK       => dsRxDataK,
         dsTxData        => dsTxData,
         dsTxDataK       => dsTxDataK,
         dsRxClk         => dsRxClk,
         dsRxRst         => dsRxRst,
         dsRxErr         => (others=>'0'),
         -- BP DS Ports
         bpTxData        => bpTxData,
         bpTxDataK       => bpTxDataK,
         bpStatus        => (others=>XPM_BP_LINK_STATUS_INIT_C),
         bpRxLinkFull    => (others=>x"0000"),
         ----------------------
         -- Top Level Interface
         ----------------------
         regclk          => regClk,
         update          => toSlv(1,NPartitions),
         status          => xpmStatus,
         config          => xpmConfig,
         -- Timing Interface (timingClk domain) 
         timingClk         => recTimingClk,
         timingRst         => recTimingRst,
         timingin          => xData,
         timingFbClk       => '0',
         timingFbRst       => '1',
         timingFbId        => (others=>'0'),
         timingFb          => open );
--         timingBus         => timingBus,
--         exptBus           => exptBus );

     busL: process( recTimingClk ) is
     begin
       if rising_edge(recTimingClk) then
         if timingBus.strobe = '1' then
           timingBusL <= timingBus;
         else
           timingBusL.strobe <= '0';
         end if;
       end if;
     end process busL;
     
end top_level_app;
