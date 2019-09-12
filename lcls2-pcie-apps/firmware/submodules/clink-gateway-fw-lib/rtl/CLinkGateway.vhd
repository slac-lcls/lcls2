-------------------------------------------------------------------------------
-- File       : CLinkGateway.vhd
-- Company    : SLAC National Accelerator Laboratory
-------------------------------------------------------------------------------
-- Description: CameraLink Gateway Core
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
use IEEE.std_logic_unsigned.all;
use IEEE.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.AxiLitePkg.all;

entity CLinkGateway is
   generic (
      TPD_G        : time                 := 1 ns;
      CHAN_COUNT_G : integer range 1 to 2 := 1;
      PGP_TYPE_G   : boolean              := false;  -- False: PGPv2b@3.125Gb/s, True: PGPv3@10.3125Gb/s, 
      BUILD_INFO_G : BuildInfoType;
      SIMULATION_G : boolean              := false);
   port (
      -- Clink Ports
      cbl0Half0P    : inout slv(4 downto 0);         --  2,  4,  5,  6, 3
      cbl0Half0M    : inout slv(4 downto 0);         -- 15, 17, 18, 19 16
      cbl0Half1P    : inout slv(4 downto 0);         --  8, 10, 11, 12,  9
      cbl0Half1M    : inout slv(4 downto 0);         -- 21, 23, 24, 25, 22
      cbl0SerP      : out   sl;         -- 20
      cbl0SerM      : out   sl;         -- 7
      cbl1Half0P    : inout slv(4 downto 0);         --  2,  4,  5,  6, 3
      cbl1Half0M    : inout slv(4 downto 0);         -- 15, 17, 18, 19 16
      cbl1Half1P    : inout slv(4 downto 0);         --  8, 10, 11, 12,  9
      cbl1Half1M    : inout slv(4 downto 0);         -- 21, 23, 24, 25, 22
      cbl1SerP      : out   sl;         -- 20
      cbl1SerM      : out   sl;         -- 7
      -- LEDs
      ledRed        : out   slv(1 downto 0);
      ledGrn        : out   slv(1 downto 0);
      ledBlu        : out   slv(1 downto 0);
      -- Boot Memory Ports
      bootCsL       : out   sl;
      bootMosi      : out   sl;
      bootMiso      : in    sl;
      -- Timing GPIO Ports
      timingClkSel  : out   sl;
      timingXbarSel : out   slv(3 downto 0);
      -- GTX Ports
      gtClkP        : in    slv(1 downto 0);
      gtClkN        : in    slv(1 downto 0);
      gtRxP         : in    slv(3 downto 0);
      gtRxN         : in    slv(3 downto 0);
      gtTxP         : out   slv(3 downto 0);
      gtTxN         : out   slv(3 downto 0);
      -- SFP Ports
      sfpScl        : inout slv(3 downto 0);
      sfpSda        : inout slv(3 downto 0);
      -- Misc Ports
      pwrScl        : inout sl;
      pwrSda        : inout sl;
      configScl     : inout sl;
      configSda     : inout sl;
      fdSerSdio     : inout sl;
      tempAlertL    : in    sl;
      vPIn          : in    sl;
      vNIn          : in    sl);
end CLinkGateway;

architecture mapping of CLinkGateway is

   constant AXIL_CLK_FREQ_C : real := 125.0E+6;  -- units of Hz

   constant NUM_AXIL_MASTERS_C : natural := 5;

   constant SYS_INDEX_C    : natural := 0;
   constant CLINK_INDEX_C  : natural := 1;
   constant TIMING_INDEX_C : natural := 2;
   constant PROM_INDEX_C   : natural := 3;
   constant PGP_INDEX_C    : natural := 4;

   constant XBAR_CONFIG_C : AxiLiteCrossbarMasterConfigArray(NUM_AXIL_MASTERS_C-1 downto 0) := genAxiLiteConfig(NUM_AXIL_MASTERS_C, x"00000000", 24, 20);

   signal axilWriteMasters : AxiLiteWriteMasterArray(NUM_AXIL_MASTERS_C-1 downto 0);
   signal axilWriteSlaves  : AxiLiteWriteSlaveArray(NUM_AXIL_MASTERS_C-1 downto 0);
   signal axilReadMasters  : AxiLiteReadMasterArray(NUM_AXIL_MASTERS_C-1 downto 0);
   signal axilReadSlaves   : AxiLiteReadSlaveArray(NUM_AXIL_MASTERS_C-1 downto 0);

   signal axilClk           : sl;
   signal axilRst           : sl;
   signal mAxilReadMasters  : AxiLiteReadMasterArray(1 downto 0);
   signal mAxilReadSlaves   : AxiLiteReadSlaveArray(1 downto 0);
   signal mAxilWriteMasters : AxiLiteWriteMasterArray(1 downto 0);
   signal mAxilWriteSlaves  : AxiLiteWriteSlaveArray(1 downto 0);

   signal refClk200MHz : sl;
   signal refRst200MHz : sl;

   signal dataMasters : AxiStreamMasterArray(1 downto 0);
   signal dataSlaves  : AxiStreamSlaveArray(1 downto 0);

   signal txUartMasters : AxiStreamMasterArray(1 downto 0);
   signal txUartSlaves  : AxiStreamSlaveArray(1 downto 0);
   signal rxUartMasters : AxiStreamMasterArray(1 downto 0);
   signal rxUartSlaves  : AxiStreamSlaveArray(1 downto 0);

   signal pgpTrigger : slv(1 downto 0);
   signal camCtrl    : Slv4Array(1 downto 0);

begin

   GEN_PGP3 : if (PGP_TYPE_G = true) generate
      U_PGP : entity work.Pgp3Phy
         generic map (
            TPD_G           => TPD_G,
            SIMULATION_G    => SIMULATION_G,
            -- CHAN_COUNT_G    => CHAN_COUNT_G,
            AXI_CLK_FREQ_G  => AXIL_CLK_FREQ_C,
            PHY_BASE_ADDR_G => XBAR_CONFIG_C(PGP_INDEX_C).baseAddr)
         port map (
            -- AXI-Lite Interface (axilClk domain)
            axilClk          => axilClk,
            axilRst          => axilRst,
            axilReadMasters  => mAxilReadMasters,
            axilReadSlaves   => mAxilReadSlaves,
            axilWriteMasters => mAxilWriteMasters,
            axilWriteSlaves  => mAxilWriteSlaves,
            -- PHY AXI-Lite Interface (axilClk domain)
            phyReadMaster    => axilReadMasters(PGP_INDEX_C),
            phyReadSlave     => axilReadSlaves(PGP_INDEX_C),
            phyWriteMaster   => axilWriteMasters(PGP_INDEX_C),
            phyWriteSlave    => axilWriteSlaves(PGP_INDEX_C),
            -- Camera Data Interface
            dataMasters      => dataMasters,
            dataSlaves       => dataSlaves,
            -- UART Interface
            txUartMasters    => txUartMasters,
            txUartSlaves     => txUartSlaves,
            rxUartMasters    => rxUartMasters,
            rxUartSlaves     => rxUartSlaves,
            -- Trigger and Link Status
            pgpTrigger       => pgpTrigger,
            -- Stable Reference IDELAY Clock and Reset
            refClk200MHz     => refClk200MHz,
            refRst200MHz     => refRst200MHz,
            -- PGP Ports
            pgpClkP          => gtClkP(0),
            pgpClkN          => gtClkN(0),
            pgpRxP           => gtRxP(1 downto 0),
            pgpRxN           => gtRxN(1 downto 0),
            pgpTxP           => gtTxP(1 downto 0),
            pgpTxN           => gtTxN(1 downto 0));
   end generate;

   GEN_PGP2b : if (PGP_TYPE_G = false) generate
      U_PGP : entity work.Pgp2bPhy
         generic map (
            TPD_G           => TPD_G,
            SIMULATION_G    => SIMULATION_G,
            -- CHAN_COUNT_G    => CHAN_COUNT_G,
            AXI_CLK_FREQ_G  => AXIL_CLK_FREQ_C,
            PHY_BASE_ADDR_G => XBAR_CONFIG_C(PGP_INDEX_C).baseAddr)
         port map (
            -- AXI-Lite Interface (axilClk domain)
            axilClk          => axilClk,
            axilRst          => axilRst,
            axilReadMasters  => mAxilReadMasters,
            axilReadSlaves   => mAxilReadSlaves,
            axilWriteMasters => mAxilWriteMasters,
            axilWriteSlaves  => mAxilWriteSlaves,
            -- PHY AXI-Lite Interface (axilClk domain)
            phyReadMaster    => axilReadMasters(PGP_INDEX_C),
            phyReadSlave     => axilReadSlaves(PGP_INDEX_C),
            phyWriteMaster   => axilWriteMasters(PGP_INDEX_C),
            phyWriteSlave    => axilWriteSlaves(PGP_INDEX_C),
            -- Camera Data Interface
            dataMasters      => dataMasters,
            dataSlaves       => dataSlaves,
            -- UART Interface
            txUartMasters    => txUartMasters,
            txUartSlaves     => txUartSlaves,
            rxUartMasters    => rxUartMasters,
            rxUartSlaves     => rxUartSlaves,
            -- Trigger and Link Status
            pgpTrigger       => pgpTrigger,
            -- Stable Reference IDELAY Clock and Reset
            refClk200MHz     => refClk200MHz,
            refRst200MHz     => refRst200MHz,
            -- PGP Ports
            pgpClkP          => gtClkP(0),
            pgpClkN          => gtClkN(0),
            pgpRxP           => gtRxP(1 downto 0),
            pgpRxN           => gtRxN(1 downto 0),
            pgpTxP           => gtTxP(1 downto 0),
            pgpTxN           => gtTxN(1 downto 0));
   end generate;

   --------------------------
   -- AXI-Lite: Crossbar Core
   --------------------------  
   U_XBAR : entity work.AxiLiteCrossbar
      generic map (
         TPD_G              => TPD_G,
         NUM_SLAVE_SLOTS_G  => 2,
         NUM_MASTER_SLOTS_G => NUM_AXIL_MASTERS_C,
         MASTERS_CONFIG_G   => XBAR_CONFIG_C)
      port map (
         axiClk           => axilClk,
         axiClkRst        => axilRst,
         sAxiWriteMasters => mAxilWriteMasters,
         sAxiWriteSlaves  => mAxilWriteSlaves,
         sAxiReadMasters  => mAxilReadMasters,
         sAxiReadSlaves   => mAxilReadSlaves,
         mAxiWriteMasters => axilWriteMasters,
         mAxiWriteSlaves  => axilWriteSlaves,
         mAxiReadMasters  => axilReadMasters,
         mAxiReadSlaves   => axilReadSlaves);

   -----------------
   -- System Modules
   -----------------
   U_FpgaSystem : entity work.FpgaSystem
      generic map (
         TPD_G           => TPD_G,
         SIMULATION_G    => SIMULATION_G,
         BUILD_INFO_G    => BUILD_INFO_G,
         AXI_CLK_FREQ_G  => AXIL_CLK_FREQ_C,
         AXI_BASE_ADDR_G => XBAR_CONFIG_C(SYS_INDEX_C).baseAddr)
      port map (
         -- AXI-Lite Interface (axilClk domain)
         axilClk         => axilClk,
         axilRst         => axilRst,
         axilReadMaster  => axilReadMasters(SYS_INDEX_C),
         axilReadSlave   => axilReadSlaves(SYS_INDEX_C),
         axilWriteMaster => axilWriteMasters(SYS_INDEX_C),
         axilWriteSlave  => axilWriteSlaves(SYS_INDEX_C),
         -- Boot Memory Ports
         bootCsL         => bootCsL,
         bootMosi        => bootMosi,
         bootMiso        => bootMiso,
         -- SFP Ports
         sfpScl          => sfpScl,
         sfpSda          => sfpSda,
         -- Misc Ports
         pwrScl          => pwrScl,
         pwrSda          => pwrSda,
         fdSerSdio       => fdSerSdio,
         tempAlertL      => tempAlertL,
         vPIn            => vPIn,
         vNIn            => vNIn);

   ------------------
   -- I2C PROM Module
   ------------------  
   U_AxiI2cEeprom : entity work.AxiI2cEeprom
      generic map (
         TPD_G          => TPD_G,
         ADDR_WIDTH_G   => 13,          -- (64kb:  ADDR_WIDTH_G = 13)
         I2C_SCL_FREQ_G => 400.0E+3,
         AXI_CLK_FREQ_G => AXIL_CLK_FREQ_C)
      port map (
         -- I2C Ports
         scl             => configScl,
         sda             => configSda,
         -- AXI-Lite Register Interface
         axilReadMaster  => axilReadMasters(PROM_INDEX_C),
         axilReadSlave   => axilReadSlaves(PROM_INDEX_C),
         axilWriteMaster => axilWriteMasters(PROM_INDEX_C),
         axilWriteSlave  => axilWriteSlaves(PROM_INDEX_C),
         -- Clocks and Resets
         axilClk         => axilClk,
         axilRst         => axilRst);

   ----------------
   -- CLink Wrapper
   ----------------
   U_CLinkWrapper : entity work.CLinkWrapper
      generic map (
         TPD_G            => TPD_G,
         CHAN_COUNT_G     => CHAN_COUNT_G,
         AXIL_BASE_ADDR_G => XBAR_CONFIG_C(CLINK_INDEX_C).baseAddr)
      port map (
         -- Clink Ports
         cbl0Half0P      => cbl0Half0P,
         cbl0Half0M      => cbl0Half0M,
         cbl0Half1P      => cbl0Half1P,
         cbl0Half1M      => cbl0Half1M,
         cbl0SerP        => cbl0SerP,
         cbl0SerM        => cbl0SerM,
         cbl1Half0P      => cbl1Half0P,
         cbl1Half0M      => cbl1Half0M,
         cbl1Half1P      => cbl1Half1P,
         cbl1Half1M      => cbl1Half1M,
         cbl1SerP        => cbl1SerP,
         cbl1SerM        => cbl1SerM,
         -- LEDs
         ledRed          => ledRed,
         ledGrn          => ledGrn,
         ledBlu          => ledBlu,
         -- Stable Reference IDELAY Clock and Reset
         refClk200MHz    => refClk200MHz,
         refRst200MHz    => refRst200MHz,
         -- Camera Control Bits
         camCtrl         => camCtrl,
         -- Camera Data Interface
         dataMasters     => dataMasters,
         dataSlaves      => dataSlaves,
         -- UART Interface
         txUartMasters   => txUartMasters,
         txUartSlaves    => txUartSlaves,
         rxUartMasters   => rxUartMasters,
         rxUartSlaves    => rxUartSlaves,
         -- Axi-Lite Interface
         axilClk         => axilClk,
         axilRst         => axilRst,
         axilReadMaster  => axilReadMasters(CLINK_INDEX_C),
         axilReadSlave   => axilReadSlaves(CLINK_INDEX_C),
         axilWriteMaster => axilWriteMasters(CLINK_INDEX_C),
         axilWriteSlave  => axilWriteSlaves(CLINK_INDEX_C));

   -----------------
   -- Trigger Module
   -----------------  
   U_Trig : entity work.TriggerTop
      generic map (
         TPD_G           => TPD_G,
         SIMULATION_G    => SIMULATION_G,
         AXIL_CLK_FREQ_G => AXIL_CLK_FREQ_C,
         AXI_BASE_ADDR_G => XBAR_CONFIG_C(TIMING_INDEX_C).baseAddr)
      port map (
         -- Trigger Input
         pgpTrigger      => pgpTrigger,
         camCtrl         => camCtrl,
         -- AXI-Lite Interface (axilClk domain)
         axilClk         => axilClk,
         axilRst         => axilRst,
         axilReadMaster  => axilReadMasters(TIMING_INDEX_C),
         axilReadSlave   => axilReadSlaves(TIMING_INDEX_C),
         axilWriteMaster => axilWriteMasters(TIMING_INDEX_C),
         axilWriteSlave  => axilWriteSlaves(TIMING_INDEX_C),
         -- Timing GPIO Ports
         timingClkSel    => timingClkSel,
         timingXbarSel   => timingXbarSel,
         -- Timing RX Ports
         timingClkP      => gtClkP(1),
         timingClkN      => gtClkN(1),
         timingRxP       => gtRxP(3 downto 2),
         timingRxN       => gtRxN(3 downto 2),
         timingTxP       => gtTxP(3 downto 2),
         timingTxN       => gtTxN(3 downto 2));

end mapping;
