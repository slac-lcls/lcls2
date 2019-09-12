-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : TimingRx.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2016-06-03
-- Last update: 2018-07-21
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description:
--   Common module to parse both LCLS-I and LCLS-II timing streams.
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
use work.TimingPkg.all;
use work.TimingExtnPkg.all;

entity TimingRx is
   generic (
      TPD_G               : time            := 1 ns;
      DEFAULT_CLK_SEL_G   : sl              := '1';
      CLKSEL_MODE_G       : string          := "SELECT"; -- "LCLSI","LCLSII"
      AXIL_ERROR_RESP_G   : slv(1 downto 0) := AXI_RESP_OK_C);
   port (
      rxClk               : in  sl;
      rxData              : in  TimingRxType;

      rxControl           : out TimingPhyControlType;
      rxStatus            : in  TimingPhyStatusType;
      
      timingClkSel        : out sl; -- '0'=LCLS1, '1'=LCLS2
      timingClkSelR       : out sl; 
      
      timingStreamUser    : out TimingStreamType;
      timingStreamPrompt  : out TimingStreamType;
      timingStreamStrobe  : out sl;
      timingStreamValid   : out sl;
      
      timingMessage       : out TimingMessageType;
      timingMessageStrobe : out sl;
      timingMessageValid  : out sl;
      timingExtn          : out TimingExtnType;
      timingExtnValid     : out sl;
      
      txClk               : in  sl;

      axilClk             : in  sl;
      axilRst             : in  sl;
      axilReadMaster      : in  AxiLiteReadMasterType;
      axilReadSlave       : out AxiLiteReadSlaveType;
      axilWriteMaster     : in  AxiLiteWriteMasterType;
      axilWriteSlave      : out AxiLiteWriteSlaveType
      );

end entity TimingRx;

architecture rtl of TimingRx is

   -------------------------------------------------------------------------------------------------
   -- axilClk Domain
   -------------------------------------------------------------------------------------------------
   type AxilRegType is record
      clkSel         : sl;
      cntRst         : sl;
      rxControl      : TimingPhyControlType;
      rxDown         : sl;
      streamNoDelay  : sl;
      messageDelay   : slv(19 downto 0);
      messageDelayRst: sl;
      axilReadSlave  : AxiLiteReadSlaveType;
      axilWriteSlave : AxiLiteWriteSlaveType;
   end record AxilRegType;

   constant AXIL_REG_INIT_C : AxilRegType := (
      clkSel         => ite(CLKSEL_MODE_G="SELECT",DEFAULT_CLK_SEL_G,
                            ite(CLKSEL_MODE_G="LCLSI",'0','1')),
      cntRst         => '0',
      rxControl      => TIMING_PHY_CONTROL_INIT_C,
      rxDown         => '0',
      streamNoDelay  => '0',
      messageDelay   => (others=>'0'),
      messageDelayRst=> '1',
      axilReadSlave  => AXI_LITE_READ_SLAVE_INIT_C,
      axilWriteSlave => AXI_LITE_WRITE_SLAVE_INIT_C);

   signal axilR   : AxilRegType := AXIL_REG_INIT_C;
   signal axilRin : AxilRegType;

   type RxRegType is record
     clkCnt : slv(3 downto 0);
     decErr : sl;
     dspErr : sl;
   end record;
   constant RX_REG_INIT_C : RxRegType := (
     clkCnt => (others=>'0'),
     decErr => '0',
     dspErr => '0' );

   signal rxR   : RxRegType := RX_REG_INIT_C;
   signal rxRin : RxRegType;
   
   signal staData            : Slv5Array(1 downto 0);
   signal staData12          : slv(4 downto 0);
   
   signal rxVersion          : Slv32Array(1 downto 0);
   signal rxVersion12        : slv(31 downto 0);

   signal stv                : slv(3 downto 0);
   signal axilRxLinkUp       : sl;
   signal axilVsnErr         : sl;
   signal axilVersion        : slv(31 downto 0);
   signal axilStatusCounters12,
          axilStatusCounters3 : SlVectorArray(3 downto 0, 31 downto 0);
   signal txClkCnt            : slv( 3 downto 0) := (others=>'0');
   signal txClkCntS           : slv(31 downto 0);
   signal rxRst               : slv( 1 downto 0);
   signal clkSelR             : sl;
   signal messageDelayR       : slv(19 downto 0);
   signal messageDelayRst     : sl;
   signal timingStreamNoDelayR: sl;
   signal rxStatusCount       : SlVectorArray(1 downto 0, 15 downto 0);
   signal timingTSEventCounter: slv(31 downto 0);
   signal timingTSEvCntGray_i : slv(31 downto 0);
   signal timingTSEvCntGray_o : Slv32Array(5 downto 0);
 
begin

  NOGEN_RxLcls1 : if CLKSEL_MODE_G = "LCLSII" generate
    timingStreamUser     <= TIMING_STREAM_INIT_C;
    timingStreamPrompt   <= TIMING_STREAM_INIT_C;
    timingStreamStrobe   <= '0';
    timingStreamValid    <= '0';
    timingTSEventCounter <= (others=>'0');
    rxVersion(0)         <= (others=>'1');
    staData  (0)         <= (others=>'0');
  end generate;
  GEN_RxLcls1 : if CLKSEL_MODE_G /= "LCLSII" generate
    U_RxLcls1 : entity work.TimingStreamRx
      generic map (
        TPD_G             => TPD_G,
        AXIL_ERROR_RESP_G => AXI_RESP_DECERR_C)
      port map (
        rxClk               => rxClk,
        rxRst               => rxRst(0),
        rxData              => rxData,
        timingMessageNoDely => timingStreamNoDelayR,
        timingMessageUser   => timingStreamUser,
        timingMessagePrompt => timingStreamPrompt,
        timingMessageStrobe => timingStreamStrobe,
        timingMessageValid  => timingStreamValid,
        timingTSEventCounter=> timingTSEventCounter,
        rxVersion           => rxVersion(0),
        staData             => staData  (0) );
  end generate;

  NOGEN_RxLcls2 : if CLKSEL_MODE_G = "LCLSI" generate
    timingMessage       <= TIMING_MESSAGE_INIT_C;
    timingMessageStrobe <= '0';
    timingMessageValid  <= '0';
    timingExtn          <= TIMING_EXTN_INIT_C;
    rxVersion(1)        <= (others=>'0');
    staData  (1)        <= (others=>'0');
  end generate;
  GEN_RxLcls2 : if CLKSEL_MODE_G /= "LCLSI" generate
    U_RxLcls2 : entity work.TimingFrameRx
      generic map (
        TPD_G             => TPD_G)   
      port map (
        rxClk               => rxClk,
        rxRst               => rxRst(1),
        rxData              => rxData,
        messageDelay        => messageDelayR,
        messageDelayRst     => messageDelayRst,
        timingMessage       => timingMessage,
        timingMessageStrobe => timingMessageStrobe,
        timingMessageValid  => timingMessageValid,
        timingExtn          => timingExtn,
        timingExtnValid     => timingExtnValid,
        rxVersion           => rxVersion(1),
        staData             => staData  (1) );
  end generate;
   
   axilComb : process (axilR, axilReadMaster, axilRst,
                       axilRxLinkUp,
                       axilVsnErr,
                       axilVersion,
                       axilStatusCounters12,
                       axilStatusCounters3,
                       rxStatusCount,
                       axilWriteMaster, txClkCntS,
                       rxStatusCount,
                       timingTSEvCntGray_o(0)) is
                 
      variable v          : AxilRegType;
      variable axilStatus : AxiLiteStatusType;

      -- Wrapper procedures to make calls cleaner.
      procedure axilSlaveRegisterW (addr : in slv; offset : in integer; reg : inout slv; cA : in boolean := false; cV : in slv := "0") is
      begin
         axiSlaveRegister(axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave, axilStatus, addr, offset, reg, cA, cV);
      end procedure;

      procedure axilSlaveRegisterR (addr : in slv; offset : in integer; reg : in slv) is
      begin
         axiSlaveRegister(axilReadMaster, v.axilReadSlave, axilStatus, addr, offset, reg);
      end procedure;

      procedure axilSlaveRegisterW (addr : in slv; offset : in integer; reg : inout sl) is
      begin
         axiSlaveRegister(axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave, axilStatus, addr, offset, reg);
      end procedure;

      procedure axilSlaveRegisterR (addr : in slv; offset : in integer; reg : in sl) is
      begin
         axiSlaveRegister(axilReadMaster, v.axilReadSlave, axilStatus, addr, offset, reg);
      end procedure;

      procedure axilSlaveDefault (
         axilResp : in slv(1 downto 0)) is
      begin
         axiSlaveDefault(axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave, axilStatus, axilResp);
      end procedure;

   begin
      -- Latch the current value
      v := axilR;
      v.axilReadSlave.rdata := (others=>'0');

      -- Determine the transaction type
      axiSlaveWaitTxn(axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave, axilStatus);

      -- Status Counters
      axilSlaveRegisterR(X"00", 0, muxSlVectorArray(axilStatusCounters12, 0));
      axilSlaveRegisterR(X"04", 0, muxSlVectorArray(axilStatusCounters12, 1));
      axilSlaveRegisterR(X"08", 0, muxSlVectorArray(axilStatusCounters12, 2));
      axilSlaveRegisterR(X"0C", 0, muxSlVectorArray(axilStatusCounters12, 3));
      axilSlaveRegisterR(X"10", 0, muxSlVectorArray(axilStatusCounters3, 0));
      axilSlaveRegisterR(X"14", 0, muxSlVectorArray(axilStatusCounters3, 1));
      axilSlaveRegisterR(X"18", 0, muxSlVectorArray(axilStatusCounters3, 2));
      axilSlaveRegisterR(X"1C", 0, muxSlVectorArray(axilStatusCounters3, 3));

      axilSlaveRegisterW(X"20", 0, v.cntRst);
      axilSlaveRegisterR(X"20", 1, axilRxLinkUp);
      axilSlaveRegisterW(X"20", 2, v.rxControl.polarity);
      axilSlaveRegisterW(X"20", 3, v.rxControl.reset);
      if (CLKSEL_MODE_G="SELECT") then
        axilSlaveRegisterW(X"20", 4, v.clkSel);
      else
        axilSlaveRegisterR(X"20", 4, v.clkSel);
      end if;
      axilSlaveRegisterW(X"20", 5, v.rxDown);
      axilSlaveRegisterW(X"20", 6, v.rxControl.bufferByRst);
      axilSlaveRegisterW(X"20", 7, v.rxControl.pllReset);
      axilSlaveRegisterR(X"20", 8, axilVsnErr);
      axilSlaveRegisterW(X"20",24, v.streamNoDelay);

      v.messageDelayRst := '0';
      axilSlaveRegisterW(X"24", 0, v.messageDelay);
      axilSlaveRegisterW(X"24",31, v.messageDelayRst);

      if v.messageDelay/=axilR.messageDelay then
        v.messageDelayRst := '1';
      end if;
      
      axilSlaveRegisterR(X"28", 0, txClkCntS);

      axilSlaveRegisterR(X"2C", 0, muxSlVectorArray(rxStatusCount,0));
      axilSlaveRegisterR(X"2C",16, muxSlVectorArray(rxStatusCount,1));
      axilSlaveRegisterR(X"30", 0, axilVersion);

      axilSlaveRegisterR(X"40", 0, timingTSEvCntGray_o(0));

      axilSlaveDefault(AXIL_ERROR_RESP_G);

      if axilRxLinkUp='0' then
        v.rxDown := '1';
      end if;
      
      --if (axilRst = '1') then
      --   v := AXIL_REG_INIT_C;
      --end if;

      axilRin <= v;

      axilReadSlave  <= axilR.axilReadSlave;
      axilWriteSlave <= axilR.axilWriteSlave;

   end process;

   axilSeq : process (axilClk) is
   begin
      if (rising_edge(axilClk)) then
         axilR <= axilRin after TPD_G;
      end if;
   end process;

   txClkCnt_seq : process (txClk) is
   begin
     if rising_edge(txClk) then
       txClkCnt <= txClkCnt+1 after TPD_G;
     end if;
   end process txClkCnt_seq;
   
   SynchronizerOneShotCnt_1 : entity work.SynchronizerOneShotCnt
     generic map (
       TPD_G          => TPD_G,
       CNT_RST_EDGE_G => true,
       CNT_WIDTH_G    => 32 )
     port map (
       dataIn       => txClkCnt(txClkCnt'left),
       rollOverEn   => '1',
       cntRst       => axilR.cntRst,
       dataOut      => open,
       cntOut       => txClkCntS,
       wrClk        => txClk,
       wrRst        => '0',
       rdClk        => axilClk,
       rdRst        => axilRst );

   axilRxLinkUp <= stv(1);
   rxVersion12  <= rxVersion(0) when clkSelR='0' else
                   rxVersion(1);
   staData12    <= staData(0) when clkSelR='0' else
                   staData(1);
   
   rxcomb : process(rxR, rxData) is
     variable v : RxRegType;
   begin
     v := rxR;
     v.clkCnt := rxR.clkCnt+1;
     v.decErr := rxData.decErr(0) or rxData.decErr(1);
     v.dspErr := rxData.dspErr(0) or rxData.dspErr(1);
     rxRin <= v;
   end process;

   SyncStatusVector_1 : entity work.SyncStatusVector
      generic map (
         TPD_G          => TPD_G,
         IN_POLARITY_G  => "1111",
         USE_DSP48_G    => "no",
         CNT_RST_EDGE_G => true,
         CNT_WIDTH_G    => 32,
         WIDTH_G        => 4 )
      port map (
         statusIn(3 downto 0)  => staData12(3 downto 0),
         cntRstIn       => axilR.cntRst,
         rollOverEnIn => "0111",
         cntOut       => axilStatusCounters12,
         wrClk        => rxClk,
         wrRst        => '0',
         rdClk        => axilClk,
         rdRst        => '0');

   SyncStatusVector_3 : entity work.SyncStatusVector
      generic map (
         TPD_G          => TPD_G,
         IN_POLARITY_G  => "1111",
         USE_DSP48_G    => "no",
         CNT_RST_EDGE_G => true,
         CNT_WIDTH_G    => 32,
         WIDTH_G        => 4 )
      port map (
         statusIn(0)  => rxR.clkCnt(rxR.clkCnt'left),
         statusIn(1)  => rxStatus.resetDone,
         statusIn(2)  => rxR.decErr,
         statusIn(3)  => rxR.dspErr,
         statusOut    => stv,
         cntRstIn     => axilR.cntRst,
         rollOverEnIn => "0001",
         cntOut       => axilStatusCounters3,
         wrClk        => rxClk,
         wrRst        => '0',
         rdClk        => axilClk,
         rdRst        => '0');

   U_Version : entity work.SynchronizerVector
     generic map ( WIDTH_G => 32 )
     port map ( clk     => axilClk,
                dataIn  => rxVersion12,
                dataOut => axilVersion );
   
   U_VsnErr : entity work.Synchronizer
     port map ( clk     => axilClk,
                dataIn  => staData12(4),
                dataOut => axilVsnErr );
   
   rxClkCnt_seq : process (rxClk) is
   begin
      if (rising_edge(rxClk)) then
         rxR <= rxRin after TPD_G;
      end if;
   end process rxClkCnt_seq;

   SyncRxRst : entity work.Synchronizer
     generic map ( TPD_G => TPD_G )
     port map ( clk     => rxClk,
                dataIn  => axilR.clkSel,
                dataOut => clkSelR );

   SyncDelayRst : entity work.Synchronizer
     generic map ( TPD_G => TPD_G )   
     port map ( clk     => rxClk,
                dataIn  => axilR.messageDelayRst,
                dataOut => messageDelayRst );

   SyncDelay : entity work.SynchronizerVector
     generic map ( TPD_G => TPD_G, WIDTH_G => axilR.messageDelay'length )
     port map ( clk     => rxClk,
                dataIn  => axilR.messageDelay,
                dataOut => messageDelayR );

   SyncStreamNoDelay : entity work.Synchronizer
     generic map ( TPD_G => TPD_G )    
     port map ( clk     => rxClk,
                dataIn  => axilR.streamNoDelay,
                dataOut => timingStreamNoDelayR );

   SyncRxStatus : entity work.SyncStatusVector
      generic map (
         TPD_G          => TPD_G,
         IN_POLARITY_G  => "11",
         CNT_WIDTH_G    => 16,
         WIDTH_G        => 2 )
      port map (
         statusIn(0)  => rxStatus.bufferByDone,
         statusIn(1)  => rxStatus.bufferByErr,
         cntRstIn     => '0',
         rollOverEnIn => "11",
         cntOut       => rxStatusCount,
         wrClk        => rxClk,
         wrRst        => '0',
         rdClk        => axilClk,
         rdRst        => '0');
     
   SyncBypassRst : entity work.Synchronizer
     generic map ( TPD_G => TPD_G )     
     port map ( clk     => rxClk,
                dataIn  => axilR.rxControl.bufferByRst,
                dataOut => rxControl.bufferByRst );
   
   -- gray encode event timestamp counter to bring into AXIL domain
   timingTSEvCntGray_i <= timingTSEventCounter xor '0' & timingTSEventCounter(31 downto 1);

   SyncTSEvCnt : entity work.SynchronizerVector
     generic map (
        TPD_G   => TPD_G,
        WIDTH_G => timingTSEvCntGray_i'length)
     port map (
        clk     => axilClk,
        dataIn  => timingTSEvCntGray_i,
        dataOut => timingTSEvCntGray_o(5));

   -- decode back to binary -- hope it's fast enough w/o pipelining
   timingTSEvCntGray_o(4) <= timingTSEvCntGray_o(5) xor x"0000" & timingTSEvCntGray_o(5)(31 downto 16);
   timingTSEvCntGray_o(3) <= timingTSEvCntGray_o(4) xor   x"00" & timingTSEvCntGray_o(4)(31 downto  8);
   timingTSEvCntGray_o(2) <= timingTSEvCntGray_o(3) xor    x"0" & timingTSEvCntGray_o(3)(31 downto  4);
   timingTSEvCntGray_o(1) <= timingTSEvCntGray_o(2) xor    "00" & timingTSEvCntGray_o(2)(31 downto  2);
   timingTSEvCntGray_o(0) <= timingTSEvCntGray_o(1) xor     '0' & timingTSEvCntGray_o(1)(31 downto  1);

   rxControl.reset    <= axilR.rxControl.reset or (axilRxLinkUp and (stv(2) or stv(3)));
   rxControl.inhibit  <= '0';
   rxControl.polarity <= axilR.rxControl.polarity;
   rxControl.pllReset <= axilR.rxControl.pllReset;
   
   rxRst(0)      <= '1' when (rxStatus.resetDone='0' or clkSelR='1') else '0';
   rxRst(1)      <= '1' when (rxStatus.resetDone='0' or clkSelR='0') else '0';
   timingClkSel  <= axilR.clkSel;
   timingClkSelR <= clkSelR;
   
end architecture rtl;

