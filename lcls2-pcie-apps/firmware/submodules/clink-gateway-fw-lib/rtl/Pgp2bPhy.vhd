-------------------------------------------------------------------------------
-- File       : Pgp2bPhy.vhd
-- Company    : SLAC National Accelerator Laboratory
-------------------------------------------------------------------------------
-- Description: Wrapper for PGPv2b communication
-------------------------------------------------------------------------------
-- This file is part of 'Camera link gateway'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'Camera link gateway', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.Pgp3Pkg.all;
use work.Pgp2bPkg.all;

library unisim;
use unisim.vcomponents.all;

entity Pgp2bPhy is
   generic (
      TPD_G           : time    := 1 ns;
      SIMULATION_G    : boolean := false;
      AXI_CLK_FREQ_G  : real    := 125.0E+6;  -- units of Hz
      PHY_BASE_ADDR_G : slv(31 downto 0));
   port (
      -- AXI-Lite Interface (axilClk domain)
      axilClk          : out sl;
      axilRst          : out sl;
      axilReadMasters  : out AxiLiteReadMasterArray(1 downto 0);
      axilReadSlaves   : in  AxiLiteReadSlaveArray(1 downto 0);
      axilWriteMasters : out AxiLiteWriteMasterArray(1 downto 0);
      axilWriteSlaves  : in  AxiLiteWriteSlaveArray(1 downto 0);
      -- PHY AXI-Lite Interface (axilClk domain)
      phyReadMaster    : in  AxiLiteReadMasterType;
      phyReadSlave     : out AxiLiteReadSlaveType;
      phyWriteMaster   : in  AxiLiteWriteMasterType;
      phyWriteSlave    : out AxiLiteWriteSlaveType;
      -- Camera Data Interface (axilClk domain)
      dataMasters      : in  AxiStreamMasterArray(1 downto 0);
      dataSlaves       : out AxiStreamSlaveArray(1 downto 0);
      -- UART Interface (axilClk domain)
      txUartMasters    : in  AxiStreamMasterArray(1 downto 0);
      txUartSlaves     : out AxiStreamSlaveArray(1 downto 0);
      rxUartMasters    : out AxiStreamMasterArray(1 downto 0);
      rxUartSlaves     : in  AxiStreamSlaveArray(1 downto 0);
      -- Trigger (axilClk domain)
      pgpTrigger       : out slv(1 downto 0);
      -- Stable Reference IDELAY Clock and Reset
      refClk200MHz     : out sl;
      refRst200MHz     : out sl;
      -- PGP Ports
      pgpClkP          : in  sl;
      pgpClkN          : in  sl;
      pgpRxP           : in  slv(1 downto 0);
      pgpRxN           : in  slv(1 downto 0);
      pgpTxP           : out slv(1 downto 0);
      pgpTxN           : out slv(1 downto 0));
end Pgp2bPhy;

architecture mapping of Pgp2bPhy is

   constant NUM_AXIL_MASTERS_C : natural := 6;

   constant XBAR_CONFIG_C : AxiLiteCrossbarMasterConfigArray(NUM_AXIL_MASTERS_C-1 downto 0) := genAxiLiteConfig(NUM_AXIL_MASTERS_C, PHY_BASE_ADDR_G, 20, 13);

   signal phyReadMasters  : AxiLiteReadMasterArray(NUM_AXIL_MASTERS_C-1 downto 0);
   signal phyReadSlaves   : AxiLiteReadSlaveArray(NUM_AXIL_MASTERS_C-1 downto 0);
   signal phyWriteMasters : AxiLiteWriteMasterArray(NUM_AXIL_MASTERS_C-1 downto 0);
   signal phyWriteSlaves  : AxiLiteWriteSlaveArray(NUM_AXIL_MASTERS_C-1 downto 0);


   signal pgpRxIn  : Pgp2bRxInArray(1 downto 0)  := (others => PGP2B_RX_IN_INIT_C);
   signal pgpRxOut : Pgp2bRxOutArray(1 downto 0) := (others => PGP2B_RX_OUT_INIT_C);

   signal pgpTxIn  : Pgp2bTxInArray(1 downto 0)  := (others => PGP2B_TX_IN_INIT_C);
   signal pgpTxOut : Pgp2bTxOutArray(1 downto 0) := (others => PGP2B_TX_OUT_INIT_C);

   signal pgpTxMasters : AxiStreamMasterArray(7 downto 0) := (others => AXI_STREAM_MASTER_INIT_C);
   signal pgpTxSlaves  : AxiStreamSlaveArray(7 downto 0)  := (others => AXI_STREAM_SLAVE_FORCE_C);
   signal pgpRxMasters : AxiStreamMasterArray(7 downto 0) := (others => AXI_STREAM_MASTER_INIT_C);
   signal pgpRxSlaves  : AxiStreamSlaveArray(7 downto 0)  := (others => AXI_STREAM_SLAVE_FORCE_C);
   signal pgpRxCtrl    : AxiStreamCtrlArray(7 downto 0)   := (others => AXI_STREAM_CTRL_UNUSED_C);

   signal pgpRefClk         : sl;
   signal pgpRefClkDiv2     : sl;
   signal pgpRefClkDiv2Bufg : sl;
   signal pgpRefClkDiv2Rst  : sl;

   signal sysClk : sl;
   signal sysRst : sl;

   signal pgpClk : sl;
   signal pgpRst : sl;

begin

   axilClk <= sysClk;
   axilRst <= sysRst;

   U_IBUFDS_GTE2 : IBUFDS_GTE2
      port map (
         I     => pgpClkP,
         IB    => pgpClkN,
         CEB   => '0',
         ODIV2 => pgpRefClkDiv2,
         O     => pgpRefClk);

   U_BUFG : BUFG
      port map (
         I => pgpRefClkDiv2,
         O => pgpRefClkDiv2Bufg);

   U_PwrUpRst : entity work.PwrUpRst
      generic map(
         TPD_G         => TPD_G,
         SIM_SPEEDUP_G => SIMULATION_G)
      port map (
         clk    => pgpRefClkDiv2Bufg,
         rstOut => pgpRefClkDiv2Rst);

   U_MMCM : entity work.ClockManager7
      generic map(
         TPD_G              => TPD_G,
         SIMULATION_G       => SIMULATION_G,
         TYPE_G             => "MMCM",
         INPUT_BUFG_G       => false,
         FB_BUFG_G          => false,
         RST_IN_POLARITY_G  => '1',
         NUM_CLOCKS_G       => 3,
         -- MMCM attributes
         BANDWIDTH_G        => "OPTIMIZED",
         CLKIN_PERIOD_G     => 6.4,     -- 156.25 MHz
         CLKFBOUT_MULT_F_G  => 8.00,    -- VCO = 1250MHz
         CLKOUT0_DIVIDE_F_G => 6.25,    -- 200 MHz = 1250MHz/6.25
         CLKOUT1_DIVIDE_G   => 10,      -- 125 MHz = 1250MHz/10
         CLKOUT2_DIVIDE_G   => 8)       -- 156.25 MHz = 1250MHz/8
      port map(
         clkIn     => pgpRefClkDiv2Bufg,
         rstIn     => pgpRefClkDiv2Rst,
         clkOut(0) => refClk200MHz,
         clkOut(1) => sysClk,
         clkOut(2) => pgpClk,
         rstOut(0) => refRst200MHz,
         rstOut(1) => sysRst,
         rstOut(2) => pgpRst);

   U_XBAR : entity work.AxiLiteCrossbar
      generic map (
         TPD_G              => TPD_G,
         NUM_SLAVE_SLOTS_G  => 1,
         NUM_MASTER_SLOTS_G => NUM_AXIL_MASTERS_C,
         MASTERS_CONFIG_G   => XBAR_CONFIG_C)
      port map (
         axiClk              => sysClk,
         axiClkRst           => sysRst,
         sAxiWriteMasters(0) => phyWriteMaster,
         sAxiWriteSlaves(0)  => phyWriteSlave,
         sAxiReadMasters(0)  => phyReadMaster,
         sAxiReadSlaves(0)   => phyReadSlave,
         mAxiWriteMasters    => phyWriteMasters,
         mAxiWriteSlaves     => phyWriteSlaves,
         mAxiReadMasters     => phyReadMasters,
         mAxiReadSlaves      => phyReadSlaves);

   GEN_VEC :
   for i in 1 downto 0 generate

      U_PGP : entity work.Pgp2bGtx7VarLat
         generic map (
            TPD_G             => TPD_G,
            -- CPLL Configurations
            TX_PLL_G          => "CPLL",
            RX_PLL_G          => "CPLL",
            CPLL_REFCLK_SEL_G => "001",
            CPLL_FBDIV_G      => 2,
            CPLL_FBDIV_45_G   => 5,
            CPLL_REFCLK_DIV_G => 1,
            -- MGT Configurations
            RXOUT_DIV_G       => 2,
            TXOUT_DIV_G       => 2,
            RX_CLK25_DIV_G    => 13,
            TX_CLK25_DIV_G    => 13,
            RXDFEXYDEN_G      => '1',
            RX_DFE_KL_CFG2_G  => x"301148AC",
            -- VC Configuration
            VC_INTERLEAVE_G   => 1)
         port map (
            -- GT Clocking
            stableClk        => pgpRefClkDiv2Bufg,
            gtCPllRefClk     => pgpRefClk,
            gtCPllLock       => open,
            gtQPllRefClk     => '0',
            gtQPllClk        => '0',
            gtQPllLock       => '1',
            gtQPllRefClkLost => '0',
            gtQPllReset      => open,
            -- GT Serial IO
            gtTxP            => pgpTxP(i),
            gtTxN            => pgpTxN(i),
            gtRxP            => pgpRxP(i),
            gtRxN            => pgpRxN(i),
            -- Tx Clocking
            pgpTxReset       => pgpRst,
            pgpTxRecClk      => open,
            pgpTxClk         => pgpClk,
            pgpTxMmcmReset   => open,
            pgpTxMmcmLocked  => '1',
            -- Rx clocking
            pgpRxReset       => pgpRst,
            pgpRxRecClk      => open,
            pgpRxClk         => pgpClk,
            pgpRxMmcmReset   => open,
            pgpRxMmcmLocked  => '1',
            -- Non VC TX Signals
            pgpTxIn          => pgpTxIn(i),
            pgpTxOut         => pgpTxOut(i),
            -- Non VC RX Signals
            pgpRxIn          => pgpRxIn(i),
            pgpRxOut         => pgpRxOut(i),
            -- Frame TX Interface
            pgpTxMasters     => pgpTxMasters(4*i+3 downto 4*i),
            pgpTxSlaves      => pgpTxSlaves(4*i+3 downto 4*i),
            -- Frame RX Interface
            pgpRxMasters     => pgpRxMasters(4*i+3 downto 4*i),
            pgpRxCtrl        => pgpRxCtrl(4*i+3 downto 4*i));

      U_SyncTrig : entity work.SynchronizerOneShot
         generic map (
            TPD_G => TPD_G)
         port map (
            clk     => sysClk,
            dataIn  => pgpRxOut(i).opCodeEn,
            dataOut => pgpTrigger(i));

      U_PgpVcWrapper : entity work.PgpVcWrapper
         generic map (
            TPD_G            => TPD_G,
            SIMULATION_G     => SIMULATION_G,
            GEN_SYNC_FIFO_G  => false,
            PHY_AXI_CONFIG_G => SSI_PGP2B_CONFIG_C)
         port map (
            -- Clocks and Resets
            sysClk          => sysClk,
            sysRst          => sysRst,
            pgpClk          => pgpClk,
            pgpRst          => pgpRst,
            -- AXI-Lite Interface (sysClk domain)
            axilReadMaster  => axilReadMasters(i),
            axilReadSlave   => axilReadSlaves(i),
            axilWriteMaster => axilWriteMasters(i),
            axilWriteSlave  => axilWriteSlaves(i),
            -- Camera Data Interface (sysClk domain)
            dataMaster      => dataMasters(i),
            dataSlave       => dataSlaves(i),
            -- UART Interface (sysClk domain)
            txUartMaster    => txUartMasters(i),
            txUartSlave     => txUartSlaves(i),
            rxUartMaster    => rxUartMasters(i),
            rxUartSlave     => rxUartSlaves(i),
            -- Frame TX Interface (pgpClk domain)
            pgpTxMasters    => pgpTxMasters(4*i+3 downto 4*i),
            pgpTxSlaves     => pgpTxSlaves(4*i+3 downto 4*i),
            -- Frame RX Interface (pgpClk domain)
            pgpRxMasters    => pgpRxMasters(4*i+3 downto 4*i),
            pgpRxCtrl       => pgpRxCtrl(4*i+3 downto 4*i),
            pgpRxSlaves     => pgpRxSlaves(4*i+3 downto 4*i));

      --------------         
      -- PGP Monitor
      --------------         
      U_PgpMon : entity work.Pgp2bAxi
         generic map (
            TPD_G              => TPD_G,
            COMMON_TX_CLK_G    => false,
            COMMON_RX_CLK_G    => false,
            WRITE_EN_G         => false,
            AXI_CLK_FREQ_G     => AXI_CLK_FREQ_G,
            STATUS_CNT_WIDTH_G => 8,
            ERROR_CNT_WIDTH_G  => 8)
         port map (
            -- TX PGP Interface (pgpTxClk)
            pgpTxClk        => pgpClk,
            pgpTxClkRst     => pgpRst,
            pgpTxIn         => pgpTxIn(i),
            pgpTxOut        => pgpTxOut(i),
            -- RX PGP Interface (pgpRxClk)
            pgpRxClk        => pgpClk,
            pgpRxClkRst     => pgpRst,
            pgpRxIn         => pgpRxIn(i),
            pgpRxOut        => pgpRxOut(i),
            -- AXI-Lite Register Interface (axilClk domain)
            axilClk         => sysClk,
            axilRst         => sysRst,
            axilReadMaster  => phyReadMasters((3*i)+0),
            axilReadSlave   => phyReadSlaves((3*i)+0),
            axilWriteMaster => phyWriteMasters((3*i)+0),
            axilWriteSlave  => phyWriteSlaves((3*i)+0));

      -----------------------------
      -- Monitor the PGP TX streams
      -----------------------------
      U_AXIS_TX_MON : entity work.AxiStreamMonAxiL
         generic map(
            TPD_G            => TPD_G,
            COMMON_CLK_G     => false,
            AXIS_CLK_FREQ_G  => 156.25E+6,
            AXIS_NUM_SLOTS_G => 4,
            AXIS_CONFIG_G    => SSI_PGP2B_CONFIG_C)
         port map(
            -- AXIS Stream Interface
            axisClk          => pgpClk,
            axisRst          => pgpRst,
            axisMasters      => pgpTxMasters(4*i+3 downto 4*i),
            axisSlaves       => pgpTxSlaves(4*i+3 downto 4*i),
            -- AXI lite slave port for register access
            axilClk          => sysClk,
            axilRst          => sysRst,
            sAxilWriteMaster => phyWriteMasters((3*i)+1),
            sAxilWriteSlave  => phyWriteSlaves((3*i)+1),
            sAxilReadMaster  => phyReadMasters((3*i)+1),
            sAxilReadSlave   => phyReadSlaves((3*i)+1));

      -----------------------------
      -- Monitor the PGP RX streams
      -----------------------------
      U_AXIS_RX_MON : entity work.AxiStreamMonAxiL
         generic map(
            TPD_G            => TPD_G,
            COMMON_CLK_G     => false,
            AXIS_CLK_FREQ_G  => 156.25E+6,
            AXIS_NUM_SLOTS_G => 4,
            AXIS_CONFIG_G    => SSI_PGP2B_CONFIG_C)
         port map(
            -- AXIS Stream Interface
            axisClk          => pgpClk,
            axisRst          => pgpRst,
            axisMasters      => pgpRxMasters(4*i+3 downto 4*i),
            axisSlaves       => (others => AXI_STREAM_SLAVE_FORCE_C),  -- SLAVE_READY_EN_G=false
            -- AXI lite slave port for register access
            axilClk          => sysClk,
            axilRst          => sysRst,
            sAxilWriteMaster => phyWriteMasters((3*i)+2),
            sAxilWriteSlave  => phyWriteSlaves((3*i)+2),
            sAxilReadMaster  => phyReadMasters((3*i)+2),
            sAxilReadSlave   => phyReadSlaves((3*i)+2));

   end generate GEN_VEC;

end mapping;
