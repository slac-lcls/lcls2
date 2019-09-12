-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : TimingBase.vhd
-- Author     : Benjamin Reese  <bareese@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-25
-- Last update: 2018-07-26
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

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.TimingExtnPkg.all;
use work.TimingPkg.all;

entity TimingBase is

   generic (
      TPD_G             : time             := 1 ns;
      TPGEN_G           : boolean          := false;
      TPGMINI_G         : boolean          := true;
      AXIL_RINGB_G      : boolean          := true;
      ASYNC_G           : boolean          := true;
      AXIL_BASE_ADDR_G  : slv(31 downto 0) := (others => '0');
      AXIL_ERROR_RESP_G : slv(1 downto 0)  := AXI_RESP_OK_C;
      USE_TPGMINI_G     : boolean          := true);
   port (

      -- Interface to GT
      gtTxUsrClk : in sl;
      gtTxUsrRst : in sl;

      gtRxRecClk    : in  sl;
      gtRxData      : in  slv(15 downto 0);
      gtRxDataK     : in  slv(1 downto 0);
      gtRxDispErr   : in  slv(1 downto 0);
      gtRxDecErr    : in  slv(1 downto 0);
      gtRxControl   : out TimingPhyControlType;
      gtRxStatus    : in  TimingPhyStatusType;
      gtTxReset     : out sl;
      gtLoopback    : out slv(2 downto 0);
      timingPhy     : out TimingPhyType;
      timingClkSel  : out sl;
      -- Decoded timing message interface
      appTimingClk  : in  sl;
      appTimingRst  : in  sl;
      appTimingBus  : out TimingBusType;
      -- Streams embedded within timing
      exptBus       : out ExptBusType;

      -- AXI Lite interface
      axilClk         : in  sl;
      axilRst         : in  sl;
      axilReadMaster  : in  AxiLiteReadMasterType;
      axilReadSlave   : out AxiLiteReadSlaveType;
      axilWriteMaster : in  AxiLiteWriteMasterType;
      axilWriteSlave  : out AxiLiteWriteSlaveType);


end entity TimingBase;

architecture rtl of TimingBase is

   constant USE_TPGMINI_C               : boolean := USE_TPGMINI_G and not TPGEN_G;
   constant FRAME_RX_AXIL_INDEX_C       : natural := 0;
   constant RAW_BUFFER_AXIL_INDEX_C     : natural := 1;
   constant MESSAGE_BUFFER_AXIL_INDEX_C : natural := 2;
   constant FRAME_TX_AXIL_INDEX_C       : natural := ite(AXIL_RINGB_G, 3, 1);

   function numAxilMasters (use_ringb : boolean; use_tpgmini : boolean) return integer is
      variable r : integer := 1;
   begin
      if (use_ringb) then
         r := r+2;
      end if;
      if (use_tpgmini) then
         r := r+1;
      end if;
      return r;
   end function;

   constant NUM_AXIL_MASTERS_C : integer := numAxilMasters(AXIL_RINGB_G, USE_TPGMINI_C);

   function axilMastersConfig (use_ringb : boolean; use_tpgmini : boolean) return AxiLiteCrossbarMasterConfigArray is
      variable config : AxiLiteCrossbarMasterConfigArray(NUM_AXIL_MASTERS_C-1 downto 0);
   begin
      config(0) := (baseAddr     => AXIL_BASE_ADDR_G + X"00000",
                    addrBits     => 16,
                    connectivity => X"FFFF");
      if (use_ringb) then
         config(1) := (baseAddr     => AXIL_BASE_ADDR_G + X"10000",
                       addrBits     => 16,
                       connectivity => X"FFFF");
         config(2) := (baseAddr     => AXIL_BASE_ADDR_G + X"20000",
                       addrBits     => 16,
                       connectivity => X"FFFF");
         if (use_tpgmini) then
            config(3) := (baseAddr     => AXIL_BASE_ADDR_G + X"30000",
                          addrBits     => 16,
                          connectivity => X"FFFF");
         end if;
      elsif (use_tpgmini) then
         config(1) := (baseAddr     => AXIL_BASE_ADDR_G + X"30000",
                       addrBits     => 16,
                       connectivity => X"FFFF");
      end if;
      return config;
   end function;

   constant AXIL_MASTERS_CONFIG_C : AxiLiteCrossbarMasterConfigArray := axilMastersConfig(AXIL_RINGB_G, USE_TPGMINI_C);

   signal locAxilWriteMasters : AxiLiteWriteMasterArray(NUM_AXIL_MASTERS_C-1 downto 0);
   signal locAxilWriteSlaves  : AxiLiteWriteSlaveArray (NUM_AXIL_MASTERS_C-1 downto 0);
   signal locAxilReadMasters  : AxiLiteReadMasterArray (NUM_AXIL_MASTERS_C-1 downto 0);
   signal locAxilReadSlaves   : AxiLiteReadSlaveArray (NUM_AXIL_MASTERS_C-1 downto 0);

   signal timingRx            : TimingRxType;
   constant TIMING_FRAME_LEN  : integer                                := TIMING_MESSAGE_BITS_C;
   signal timingStrobe        : sl;
   signal timingValid         : sl                                     := '1';
   signal timingMessageStrobe : sl;
   signal timingMessageValid  : sl                                     := '1';
   signal timingStreamStrobe  : sl;
   signal timingStreamValid   : sl                                     := '1';
   signal timingMessage       : TimingMessageType;
   signal timingStream        : TimingStreamType;
   signal timingFrameSlv      : slv(TIMING_FRAME_LEN-1 downto 0);
   signal appTimingFrameSlv   : slv(TIMING_FRAME_LEN-1 downto 0);
   signal timingFrameSlvShift : slv(TIMING_FRAME_LEN+31 downto 0)      := (others=>'0');
   signal timingFrameSlvValid : slv((TIMING_FRAME_LEN+31)/32 downto 0) := (others=>'0');

   signal clkSel              : sl;
   signal clkSelTx            : sl;
   signal timingClkSelR       : sl;
   signal timingClkSelApp     : sl;

   signal linkUpV1            : sl;
   signal linkUpV2            : sl;
   
   signal itxData             : Slv16Array(1 downto 0);
   signal itxDataK            : Slv2Array (1 downto 0);
begin

   AxiLiteCrossbar_1 : entity work.AxiLiteCrossbar
      generic map (
         TPD_G              => TPD_G,
         NUM_SLAVE_SLOTS_G  => 1,
         NUM_MASTER_SLOTS_G => NUM_AXIL_MASTERS_C,
         DEC_ERROR_RESP_G   => AXI_RESP_DECERR_C,
         MASTERS_CONFIG_G   => AXIL_MASTERS_CONFIG_C,
         DEBUG_G            => true)
      port map (
         axiClk              => axilClk,
         axiClkRst           => axilRst,
         sAxiWriteMasters(0) => axilWriteMaster,
         sAxiWriteSlaves(0)  => axilWriteSlave,
         sAxiReadMasters(0)  => axilReadMaster,
         sAxiReadSlaves(0)   => axilReadSlave,
         mAxiWriteMasters    => locAxilWriteMasters,
         mAxiWriteSlaves     => locAxilWriteSlaves,
         mAxiReadMasters     => locAxilReadMasters,
         mAxiReadSlaves      => locAxilReadSlaves);

   -------------------------------------------------------------------------------------------------
   -- Receive and decode timing data frames from GT
   -------------------------------------------------------------------------------------------------
   timingRx.data   <= gtRxData;
   timingRx.dataK  <= gtRxDataK;
   timingRx.decErr <= gtRxDecErr;
   timingRx.dspErr <= gtRxDispErr;
   timingClkSel    <= clkSel;
   
   U_TimingRx : entity work.TimingRx
      generic map (
         TPD_G             => TPD_G,
         CLKSEL_MODE_G     => CLKSEL_MODE_G,
         AXIL_ERROR_RESP_G => AXI_RESP_DECERR_C)
      port map (
         txClk               => gtTxUsrClk,
         rxClk               => gtRxRecClk,
         rxStatus            => gtRxStatus,
         rxControl           => gtRxControl,
         rxData              => timingRx,
         timingClkSel        => clkSel,
         timingClkSelR       => timingClkSelR,
         timingStream        => timingStream,
         timingStreamStrobe  => timingStreamStrobe,
         timingStreamValid   => timingStreamValid,
         timingMessage       => timingMessage,
         timingMessageStrobe => timingMessageStrobe,
         timingMessageValid  => timingMessageValid,
         exptMessage         => exptBus.message,
         exptMessageValid    => exptBus.valid,
         axilClk             => axilClk,
         axilRst             => axilRst,
         axilReadMaster      => locAxilReadMasters (FRAME_RX_AXIL_INDEX_C),
         axilReadSlave       => locAxilReadSlaves (FRAME_RX_AXIL_INDEX_C),
         axilWriteMaster     => locAxilWriteMasters(FRAME_RX_AXIL_INDEX_C),
         axilWriteSlave      => locAxilWriteSlaves (FRAME_RX_AXIL_INDEX_C));

   GEN_AXIL_RINGB : if AXIL_RINGB_G generate
      -------------------------------------------------------------------------------------------------
      -- Ring buffer to log raw GT words
      -------------------------------------------------------------------------------------------------
      AxiLiteRingBuffer_1 : entity work.AxiLiteRingBuffer
         generic map (
            TPD_G            => TPD_G,
            BRAM_EN_G        => true,
            REG_EN_G         => true,
            DATA_WIDTH_G     => 18,
            RAM_ADDR_WIDTH_G => 13)
         port map (
            dataClk                 => gtRxRecClk,
            dataRst                 => '0',
            dataValid               => '1',
            dataValue(15 downto 0)  => gtRxData,
            dataValue(17 downto 16) => gtRxDataK,
            axilClk                 => axilClk,
            axilRst                 => axilRst,
            axilReadMaster          => locAxilReadMasters (RAW_BUFFER_AXIL_INDEX_C),
            axilReadSlave           => locAxilReadSlaves (RAW_BUFFER_AXIL_INDEX_C),
            axilWriteMaster         => locAxilWriteMasters(RAW_BUFFER_AXIL_INDEX_C),
            axilWriteSlave          => locAxilWriteSlaves (RAW_BUFFER_AXIL_INDEX_C));

      -------------------------------------------------------------------------------------------------
      -- Ring buffer to log received timing messages
      -------------------------------------------------------------------------------------------------
      process (gtRxRecClk) is
      begin
         if rising_edge(gtRxRecClk) then
            if timingStrobe = '1' then
               timingFrameSlvShift <= timingFrameSlv & x"deadbeef" after TPD_G;
               timingFrameSlvValid <= (others => '1') after TPD_G;
            else
               timingFrameSlvShift <= x"00000000" & timingFrameSlvShift(timingFrameSlvShift'left downto 32) after TPD_G;
               timingFrameSlvValid <= '0' & timingFrameSlvValid(timingFrameSlvValid'left downto 1) after TPD_G;
            end if;
         end if;
      end process;

      AxiLiteRingBuffer_2 : entity work.AxiLiteRingBuffer
         generic map (
            TPD_G            => TPD_G,
            BRAM_EN_G        => true,
            REG_EN_G         => true,
            DATA_WIDTH_G     => 32,
            RAM_ADDR_WIDTH_G => 13)
         port map (
            dataClk         => gtRxRecClk,
            dataRst         => '0',
            dataValid       => timingFrameSlvValid(0),
            dataValue       => timingFrameSlvShift(31 downto 0),
            axilClk         => axilClk,
            axilRst         => axilRst,
            axilReadMaster  => locAxilReadMasters (MESSAGE_BUFFER_AXIL_INDEX_C),
            axilReadSlave   => locAxilReadSlaves (MESSAGE_BUFFER_AXIL_INDEX_C),
            axilWriteMaster => locAxilWriteMasters(MESSAGE_BUFFER_AXIL_INDEX_C),
            axilWriteSlave  => locAxilWriteSlaves (MESSAGE_BUFFER_AXIL_INDEX_C));
   end generate;

   timingPhy.control.reset       <= '0';
   timingPhy.control.bufferByRst <= '0';
   
   GEN_MINICORE : if USE_TPGMINI_C generate
      TPGMiniCore_1 : entity work.TPGMiniCore
         generic map (
            TPD_G      => TPD_G, 
            NARRAYSBSA => 2)
         port map (
            txClk          => gtTxUsrClk,
            txRst          => gtTxUsrRst,
            txRdy          => '1',
            txData         => itxData,
            txDataK        => itxDataK,
            txPolarity     => timingPhy.control.polarity,
            txResetO       => gtTxReset,
            txLoopback     => gtLoopback,
            txInhibit      => timingPhy.control.inhibit,
            axiClk         => axilClk,
            axiRst         => axilRst,
            axiReadMaster  => locAxilReadMasters (FRAME_TX_AXIL_INDEX_C),
            axiReadSlave   => locAxilReadSlaves (FRAME_TX_AXIL_INDEX_C),
            axiWriteMaster => locAxilWriteMasters(FRAME_TX_AXIL_INDEX_C),
            axiWriteSlave  => locAxilWriteSlaves (FRAME_TX_AXIL_INDEX_C));

      U_SyncClkSel : entity work.Synchronizer
         generic map (TPD_G=> TPD_G)      
        port map ( clk     => gtTxUsrClk,
                   dataIn  => clkSel,
                   dataOut => clkSelTx );

      timingPhy.data  <= itxData(0) when clkSelTx='0' else
                         itxData(1);
      timingPhy.dataK <= itxDataK(0) when clkSelTx='0' else
                         itxDataK(1);
                        
   end generate GEN_MINICORE;

   NOGEN_MINICORE : if not USE_TPGMINI_C generate
      timingPhy.data     <= (others => '0');
      timingPhy.dataK    <= "00";
      timingPhy.control.polarity <= '0';
      timingPhy.control.inhibit  <= '0';
      gtLoopback         <= "000";
   end generate NOGEN_MINICORE;

   -------------------------------------------------------------------------------------------------
   -- Synchronize timing message to appTimingClk
   -------------------------------------------------------------------------------------------------


   timingFrameSlv <= toSlv(timingMessage) when timingClkSelR = '1' else
                     (slvZero(TIMING_FRAME_LEN-TIMING_STREAM_BITS_C) & toSlv(timingStream));
   timingStrobe   <= timingMessageStrobe  when timingClkSelR='1' else
                     timingStreamStrobe;
   timingValid    <= timingMessageValid   when timingClkSelR='1' else
                     timingStreamValid;
   
   GEN_ASYNC: if ASYNC_G generate
     process (timingClkSelApp, appTimingFrameSlv) is
     begin
       if timingClkSelApp='0' then
         appTimingBus.stream  <= toTimingStreamType(appTimingFrameSlv(TIMING_STREAM_BITS_C-1 downto 0));
         appTimingBus.message <= TIMING_MESSAGE_INIT_C;
       else
         appTimingBus.message <= toTimingMessageType(appTimingFrameSlv(TIMING_MESSAGE_BITS_C-1 downto 0));
         appTimingBus.stream  <= TIMING_STREAM_INIT_C;
       end if;
     end process;

     -- Need to syncrhonize timingClkSelR to appTimingClk so we can use
     -- it to switch between stream and message in appTimingClk domain
     U_Synchronizer_1 : entity work.Synchronizer
       generic map (
         TPD_G => TPD_G)
       port map (
         clk     => appTimingClk,       -- [in]
         rst     => appTimingRst,       -- [in]
         dataIn  => timingClkSelR,      -- [in]
         dataOut => timingClkSelApp);   -- [out]

      SynchronizerFifo_1 : entity work.SynchronizerFifo
         generic map (
            TPD_G        => TPD_G,
            DATA_WIDTH_G => TIMING_FRAME_LEN+1)
         port map (
            rst                             => appTimingRst,
            wr_clk                          => gtRxRecClk,
            wr_en                           => timingStrobe,
            din(TIMING_FRAME_LEN downto 1)  => timingFrameSlv,
            din(0)                          => timingValid,
            rd_clk                          => appTimingClk,
            dout(TIMING_FRAME_LEN downto 1) => appTimingFrameSlv,
            dout(0)                         => appTimingBus.valid,
            valid                           => appTimingBus.strobe);
   end generate;

   NO_GEN_ASYNC : if not ASYNC_G generate
      appTimingBus.stream  <= timingStream;
      appTimingBus.message <= timingMessage;
      appTimingBus.strobe  <= timingStrobe;
      appTimingBus.valid   <= timingValid;
   end generate;

   U_SYNC_LinkV1 : entity work.Synchronizer
     generic map (TPD_G => TPD_G)   
     port map ( clk     => appTimingClk,
                dataIn  => linkUpV1,
                dataOut => appTimingBus.v1.linkUp );
   
   appTimingBus.v1.gtRxData    <= gtRxData    when(timingClkSelR = '0') else (others=>'0');
   appTimingBus.v1.gtRxDataK   <= gtRxDataK   when(timingClkSelR = '0') else (others=>'0');
   appTimingBus.v1.gtRxDispErr <= gtRxDispErr when(timingClkSelR = '0') else (others=>'0');
   appTimingBus.v1.gtRxDecErr  <= gtRxDecErr  when(timingClkSelR = '0') else (others=>'0');

   U_SYNC_LinkV2 : entity work.Synchronizer
     generic map (TPD_G => TPD_G)     
     port map ( clk     => appTimingClk,
                dataIn  => linkUpV2,
                dataOut => appTimingBus.v2.linkUp );
   
   linkUpV1 <= gtRxStatus.locked and not timingClkSelR;
   linkUpV2 <= gtRxStatus.locked and timingClkSelR;

end rtl;
