-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiPgp.vhd
-- Author     : Matt Weaver
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-10-30
-- Last update: 2017-11-14
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Common Carrier Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Common Carrier Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;
use work.Pgp2bPkg.all;

library unisim;
use unisim.vcomponents.all;

entity DtiPgp is
   generic (
      TPD_G            : time            := 1 ns;
      AXI_ERROR_RESP_G : slv(1 downto 0) := AXI_RESP_DECERR_C);
   port (
      -- PGP Ports
      pgpRxP         : in  sl;
      pgpRxN         : in  sl;
      pgpTxP         : out sl;
      pgpTxN         : out sl;
      pgpClkP        : in  sl;
      pgpClkN        : in  sl;
      -- AXI-Lite Interface
      axilClk           : in  sl;
      axilRst           : in  sl;
      axilReadMaster    : in  AxiLiteReadMasterType;
      axilReadSlave     : out AxiLiteReadSlaveType;
      axilWriteMaster   : in  AxiLiteWriteMasterType;
      axilWriteSlave    : out AxiLiteWriteSlaveType;
      -- AXI Stream interface
      obMasters      : in  AxiStreamMasterArray(3 downto 0);
      obSlaves       : out AxiStreamSlaveArray(3 downto 0);
      ibMasters      : out AxiStreamMasterArray(3 downto 0);
      ibSlaves       : in  AxiStreamSlaveArray(3 downto 0) );
end DtiPgp;


architecture mapping of DtiPgp is

   -------------------------------------------------------------------------------------------------
   -- PGP Signals
   -------------------------------------------------------------------------------------------------
   signal pgpTxIn       : Pgp2bTxInType;
   signal pgpTxOut      : Pgp2bTxOutType;
   signal pgpRxIn       : Pgp2bRxInType;
   signal pgpRxOut      : Pgp2bRxOutType;
   signal pgpTxMasters  : AxiStreamMasterArray(3 downto 0) := (others => AXI_STREAM_MASTER_INIT_C);
   signal pgpTxSlaves   : AxiStreamSlaveArray(3 downto 0)  := (others => AXI_STREAM_SLAVE_FORCE_C);
   signal pgpRxMasters  : AxiStreamMasterArray(3 downto 0) := (others => AXI_STREAM_MASTER_INIT_C);
   signal pgpRxCtrl     : AxiStreamCtrlArray(3 downto 0)   := (others => AXI_STREAM_CTRL_UNUSED_C);
   signal pgpRefClkDiv2 : sl;
   signal pgpRefClk     : sl;
   signal pgpClk        : sl;
   signal pgpRst        : sl;

   -------------------------------------------------------------------------------------------------
   -- Packetizer constants and signals
   -------------------------------------------------------------------------------------------------
   -- This should really go in a AxiStreamPacketizerPkg
   constant IB_PACKETIZER_AXIS_CONFIG_C : AxiStreamConfigType := (
      TSTRB_EN_C    => false,
      TDATA_BYTES_C => 8,
      TDEST_BITS_C  => 8,
      TID_BITS_C    => 8,
      TKEEP_MODE_C  => TKEEP_COMP_C,
      TUSER_BITS_C  => 8,
      TUSER_MODE_C  => TUSER_FIRST_LAST_C);

   constant OB_PACKETIZER_AXIS_CONFIG_C : AxiStreamConfigType :=
      ssiAxiStreamConfig(
         dataBytes => 8,
         tKeepMode => TKEEP_COMP_C,
         tUserMode => TUSER_FIRST_LAST_C,
         tDestBits => 0,
         tUserBits => 2);

   signal ibPacketizerMaster : AxiStreamMasterType;
   signal ibPacketizerSlave : AxiStreamSlaveType;   
   signal obPacketizerMaster : AxiStreamMasterType;
   signal obPacketizerSlave : AxiStreamSlaveType;   


begin

   bpMsgBus    <= (others => BP_MSG_BUS_INIT_C);
   bpMsgSlaves <= (others => AXI_STREAM_SLAVE_FORCE_C);


   mAxilReadMasters(1)  <= AXI_LITE_READ_MASTER_INIT_C;
   mAxilWriteMasters(1) <= AXI_LITE_WRITE_MASTER_INIT_C;

   U_IBUFDS_GTE3 : IBUFDS_GTE3
      generic map (
         REFCLK_EN_TX_PATH  => '0',
         REFCLK_HROW_CK_SEL => "00",    -- 2'b00: ODIV2 = O
         REFCLK_ICNTL_RX    => "00")
      port map (
         I     => pgpClkP,
         IB    => pgpClkN,
         CEB   => '0',
         ODIV2 => pgpRefClkDiv2,        -- Divide by 1
         O     => pgpRefClk);

   U_BUFG_GT : BUFG_GT
      port map (
         I       => pgpRefClkDiv2,
         CE      => '1',
         CLR     => '0',
         CEMASK  => '1',
         CLRMASK => '1',
         DIV     => "000",              -- Divide by 1
         O       => pgpClk);

   U_PwrUpRst : entity work.PwrUpRst
      generic map (
         TPD_G          => TPD_G,
         SIM_SPEEDUP_G  => SIMULATION_G,
         IN_POLARITY_G  => '1',
         OUT_POLARITY_G => '1')
      port map (
         clk    => pgpClk,
         rstOut => pgpRst);


   REAL_PGP : if (not SIMULATION_G) generate

      Pgp2bGthUltra_1 : entity work.DebugRtmPgp2bGthUltra
         generic map (
            TPD_G             => TPD_G,
            PAYLOAD_CNT_TOP_G => 7,
            VC_INTERLEAVE_G   => 0,
            NUM_VC_EN_G       => 2)
         port map (
            stableClk        => axilClk,
            stableRst        => axilRst,
            gtRefClk         => pgpRefClk,
            pgpGtTxP         => pgpTxP,
            pgpGtTxN         => pgpTxN,
            pgpGtRxP         => pgpRxP,
            pgpGtRxN         => pgpRxN,
            pgpTxReset       => pgpRst,
            pgpTxRecClk      => open,
            pgpTxClk         => pgpClk,
            pgpTxMmcmLocked  => '1',
            pgpRxReset       => pgpRst,
            pgpRxRecClk      => open,
            pgpRxClk         => pgpClk,
            pgpRxMmcmLocked  => '1',
            pgpRxIn          => pgpRxIn,
            pgpRxOut         => pgpRxOut,
            pgpTxIn          => pgpTxIn,
            pgpTxOut         => pgpTxOut,
            pgpTxMasters     => pgpTxMasters,
            pgpTxSlaves      => pgpTxSlaves,
            pgpRxMasters     => pgpRxMasters,
            pgpRxMasterMuxed => open,
            pgpRxCtrl        => pgpRxCtrl);

   end generate REAL_PGP;

   SIM_PGP : if (SIMULATION_G) generate
      PgpSimModel_1 : entity work.PgpSimModel
         generic map (
            TPD_G      => TPD_G,
            LANE_CNT_G => 2)
         port map (
            pgpTxClk         => pgpClk,
            pgpTxClkRst      => pgpRst,
            pgpTxIn          => pgpTxIn,
            pgpTxOut         => pgpTxOut,
            pgpTxMasters     => pgpTxMasters,
            pgpTxSlaves      => pgpTxSlaves,
            pgpRxClk         => pgpClk,
            pgpRxClkRst      => pgpRst,
            pgpRxIn          => pgpRxIn,
            pgpRxOut         => pgpRxOut,
            pgpRxMasters     => pgpRxMasters,
            pgpRxMasterMuxed => open,
            pgpRxCtrl        => pgpRxCtrl);
   end generate SIM_PGP;

   -------------------------------------------------------------------------------------------------
   -- AXIL Interface to PGP debug
   -------------------------------------------------------------------------------------------------
   Pgp2bAxi_1 : entity work.Pgp2bAxi
      generic map (
         TPD_G              => TPD_G,
         COMMON_TX_CLK_G    => false,
         COMMON_RX_CLK_G    => false,
         WRITE_EN_G         => false,
         AXI_CLK_FREQ_G     => 156.25E+6,
         STATUS_CNT_WIDTH_G => 32,
         ERROR_CNT_WIDTH_G  => 16)
      port map (
         pgpTxClk        => pgpClk,
         pgpTxClkRst     => pgpRst,
         pgpTxIn         => pgpTxIn,
         pgpTxOut        => pgpTxOut,
         pgpRxClk        => pgpClk,
         pgpRxClkRst     => pgpRst,
         pgpRxIn         => pgpRxIn,
         pgpRxOut        => pgpRxOut,
         axilClk         => axilClk,
         axilRst         => axilRst,
         axilReadMaster  => axilReadMaster,
         axilReadSlave   => axilReadSlave,
         axilWriteMaster => axilWriteMaster,
         axilWriteSlave  => axilWriteSlave);

   --------------------------------------------------
   -- Legacy AXI-Lite Master on VC0
   --------------------------------------------------
   U_SRPv0 : entity work.SrpV0AxiLite
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => false,
         EN_32BIT_ADDR_G     => true,
         BRAM_EN_G           => true,
         GEN_SYNC_FIFO_G     => true,
         AXI_STREAM_CONFIG_G => SSI_PGP2B_CONFIG_C)
      port map (
         -- Streaming Slave (Rx) Interface (sAxisClk domain) 
         sAxisClk            => pgpClk,
         sAxisRst            => pgpRst,
         sAxisMaster         => pgpRxMasters(0),
         sAxisSlave          => open,
         sAxisCtrl           => pgpRxCtrl(0),
         -- Streaming Master (Tx) Data Interface (mAxisClk domain)
         mAxisClk            => pgpClk,
         mAxisRst            => pgpRst,
         mAxisMaster         => pgpTxMasters(0),
         mAxisSlave          => pgpTxSlaves(0),
         -- AXI Lite Bus (axiLiteClk domain)
         axiLiteClk          => axilClk,
         axiLiteRst          => axilRst,
         mAxiLiteReadMaster  => mAxilReadMasters(0),
         mAxiLiteReadSlave   => mAxilReadSlaves(0),
         mAxiLiteWriteMaster => mAxilWriteMasters(0),
         mAxiLiteWriteSlave  => mAxilWriteSlaves(0));

   -------------------------------------------------------------------------------------------------
   -- Tie Off unused BSA and PGP streams
   -------------------------------------------------------------------------------------------------
   pgpRxCtrl(3 downto 1)    <= (others => AXI_STREAM_CTRL_UNUSED_C);
   pgpTxMasters(3 downto 2) <= (others => AXI_STREAM_MASTER_INIT_C);

   obSlaves(2 downto 0)  <= (others => AXI_STREAM_SLAVE_FORCE_C);
   ibMasters(3 downto 0) <= (others => AXI_STREAM_MASTER_INIT_C);

   -------------------------------------------------------------------------------------------------
   -- Transition the BSA waveform data stream into packetizer input format
   -- Convert to PGP clock
   -------------------------------------------------------------------------------------------------
   AxiStreamFifo_PACKETIZER : entity work.AxiStreamFifoV2
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => true,
         BRAM_EN_G           => false,
         XIL_DEVICE_G        => "ULTRASCALE",
         GEN_SYNC_FIFO_G     => false,
         INT_PIPE_STAGES_G   => 0,
         PIPE_STAGES_G       => 1,
         FIFO_ADDR_WIDTH_G   => 4,
         SLAVE_AXI_CONFIG_G  => ETH_AXIS_CONFIG_C,
         MASTER_AXI_CONFIG_G => IB_PACKETIZER_AXIS_CONFIG_C)
      port map (
         sAxisClk    => axilClk,
         sAxisRst    => axilRst,
         sAxisMaster => obMasters(3),
         sAxisSlave  => obSlaves(3),
         sAxisCtrl   => open,
         mAxisClk    => pgpClk,
         mAxisRst    => pgpRst,
         mAxisMaster => ibPacketizerMaster,
         mAxisSlave  => ibPacketizerSlave);

   -------------------------------------------------------------------------------------------------
   -- Packetize the stream
   --------------------------------------------------------------------------------------------------
   U_AxiStreamPacketizer_1 : entity work.AxiStreamPacketizer
      generic map (
         TPD_G                => TPD_G,
         MAX_PACKET_BYTES_G   => 4112,
         MIN_TKEEP_G          => X"000F",
         INPUT_PIPE_STAGES_G  => 0,
         OUTPUT_PIPE_STAGES_G => 1)
      port map (
         axisClk     => pgpClk,              -- [in]
         axisRst     => pgpRst,              -- [in]
         sAxisMaster => ibPacketizerMaster,  -- [in]
         sAxisSlave  => ibPacketizerSlave,   -- [out]
         mAxisMaster => obPacketizerMaster,  -- [out]
         mAxisSlave  => obPacketizerSlave);  -- [in]

   -------------------------------------------------------------------------------------------------
   -- Convert the stream to pgp format (16-bit)
   -------------------------------------------------------------------------------------------------
   AxiStreamFifo_PGP : entity work.AxiStreamFifoV2
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => true,
         VALID_THOLD_G       => 1,
         BRAM_EN_G           => true,
         XIL_DEVICE_G        => "ULTRASCALE",
         USE_BUILT_IN_G      => false,
         GEN_SYNC_FIFO_G     => true,
         CASCADE_SIZE_G      => 1,
         FIFO_ADDR_WIDTH_G   => 9,
         FIFO_FIXED_THRESH_G => true,
         FIFO_PAUSE_THRESH_G => 2**9-1,
         SLAVE_AXI_CONFIG_G  => OB_PACKETIZER_AXIS_CONFIG_C,
         MASTER_AXI_CONFIG_G => SSI_PGP2B_CONFIG_C)
      port map (
         sAxisClk    => pgpClk,
         sAxisRst    => pgpRst,
         sAxisMaster => obPacketizerMaster,
         sAxisSlave  => obPacketizerSlave,
         mAxisClk    => pgpClk,
         mAxisRst    => pgpRst,
         mAxisMaster => pgpTxMasters(1),
         mAxisSlave  => pgpTxSlaves(1));

end mapping;
