-------------------------------------------------------------------------------
-- File       : PgpVcWrapper.vhd
-- Company    : SLAC National Accelerator Laboratory
-------------------------------------------------------------------------------
-- Description: PGP Virtual Channel Mapping
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
use work.Pgp3Pkg.all;

entity PgpVcWrapper is
   generic (
      TPD_G            : time    := 1 ns;
      SIMULATION_G     : boolean := false;
      GEN_SYNC_FIFO_G  : boolean := false;
      PHY_AXI_CONFIG_G : AxiStreamConfigType);
   port (
      -- Clocks and Resets
      sysClk          : in  sl;
      sysRst          : in  sl;
      pgpClk          : in  sl;
      pgpRst          : in  sl;
      -- AXI-Lite Interface (sysClk domain)
      axilReadMaster  : out AxiLiteReadMasterType;
      axilReadSlave   : in  AxiLiteReadSlaveType;
      axilWriteMaster : out AxiLiteWriteMasterType;
      axilWriteSlave  : in  AxiLiteWriteSlaveType;
      -- Camera Data Interface (sysClk domain)
      dataMaster      : in  AxiStreamMasterType;
      dataSlave       : out AxiStreamSlaveType;
      -- UART Interface (sysClk domain)
      txUartMaster    : in  AxiStreamMasterType;
      txUartSlave     : out AxiStreamSlaveType;
      rxUartMaster    : out AxiStreamMasterType;
      rxUartSlave     : in  AxiStreamSlaveType;
      -- Frame TX Interface (pgpClk domain)
      pgpTxMasters    : out AxiStreamMasterArray(3 downto 0) := (others => AXI_STREAM_MASTER_INIT_C);
      pgpTxSlaves     : in  AxiStreamSlaveArray(3 downto 0);
      -- Frame RX Interface (pgpClk domain)
      pgpRxMasters    : in  AxiStreamMasterArray(3 downto 0);
      pgpRxCtrl       : out AxiStreamCtrlArray(3 downto 0)   := (others => AXI_STREAM_CTRL_UNUSED_C);
      pgpRxSlaves     : out AxiStreamSlaveArray(3 downto 0)  := (others => AXI_STREAM_SLAVE_FORCE_C));
end PgpVcWrapper;

architecture mapping of PgpVcWrapper is

begin

   U_Vc0 : entity work.SrpV3AxiLite
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => SIMULATION_G,
         GEN_SYNC_FIFO_G     => GEN_SYNC_FIFO_G,
         AXI_STREAM_CONFIG_G => PHY_AXI_CONFIG_G)
      port map (
         -- Streaming Slave (Rx) Interface (sAxisClk domain) 
         sAxisClk         => pgpClk,
         sAxisRst         => pgpRst,
         sAxisMaster      => pgpRxMasters(0),
         sAxisSlave       => pgpRxSlaves(0),
         sAxisCtrl        => pgpRxCtrl(0),
         -- Streaming Master (Tx) Data Interface (mAxisClk domain)
         mAxisClk         => pgpClk,
         mAxisRst         => pgpRst,
         mAxisMaster      => pgpTxMasters(0),
         mAxisSlave       => pgpTxSlaves(0),
         -- Master AXI-Lite Interface (axilClk domain)
         axilClk          => sysClk,
         axilRst          => sysRst,
         mAxilReadMaster  => axilReadMaster,
         mAxilReadSlave   => axilReadSlave,
         mAxilWriteMaster => axilWriteMaster,
         mAxilWriteSlave  => axilWriteSlave);

   U_Vc1_Tx : entity work.AxiStreamFifoV2
      generic map (
         -- General Configurations
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => true,
         VALID_THOLD_G       => 256,
         VALID_BURST_MODE_G  => true,
         -- FIFO configurations
         GEN_SYNC_FIFO_G     => GEN_SYNC_FIFO_G,
         FIFO_ADDR_WIDTH_G   => 9,
         -- AXI Stream Port Configurations
         SLAVE_AXI_CONFIG_G  => PGP3_AXIS_CONFIG_C,
         MASTER_AXI_CONFIG_G => PHY_AXI_CONFIG_G)
      port map (
         -- Slave Port
         sAxisClk    => sysClk,
         sAxisRst    => sysRst,
         sAxisMaster => dataMaster,
         sAxisSlave  => dataSlave,
         -- Master Port
         mAxisClk    => pgpClk,
         mAxisRst    => pgpRst,
         mAxisMaster => pgpTxMasters(1),
         mAxisSlave  => pgpTxSlaves(1));

   U_Vc2_Tx : entity work.AxiStreamFifoV2
      generic map (
         -- General Configurations
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => true,
         VALID_THOLD_G       => 256,
         VALID_BURST_MODE_G  => true,
         -- FIFO configurations
         GEN_SYNC_FIFO_G     => GEN_SYNC_FIFO_G,
         FIFO_ADDR_WIDTH_G   => 9,
         -- AXI Stream Port Configurations
         SLAVE_AXI_CONFIG_G  => PGP3_AXIS_CONFIG_C,
         MASTER_AXI_CONFIG_G => PHY_AXI_CONFIG_G)
      port map (
         -- Slave Port
         sAxisClk    => sysClk,
         sAxisRst    => sysRst,
         sAxisMaster => txUartMaster,
         sAxisSlave  => txUartSlave,
         -- Master Port
         mAxisClk    => pgpClk,
         mAxisRst    => pgpRst,
         mAxisMaster => pgpTxMasters(2),
         mAxisSlave  => pgpTxSlaves(2));

   U_Vc2_Rx : entity work.AxiStreamFifoV2
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => SIMULATION_G,
         GEN_SYNC_FIFO_G     => GEN_SYNC_FIFO_G,
         FIFO_ADDR_WIDTH_G   => 9,
         FIFO_FIXED_THRESH_G => true,
         FIFO_PAUSE_THRESH_G => 128,
         SLAVE_AXI_CONFIG_G  => PHY_AXI_CONFIG_G,
         MASTER_AXI_CONFIG_G => PGP3_AXIS_CONFIG_C)
      port map (
         -- Slave Port
         sAxisClk    => pgpClk,
         sAxisRst    => pgpRst,
         sAxisMaster => pgpRxMasters(2),
         sAxisSlave  => pgpRxSlaves(2),
         sAxisCtrl   => pgpRxCtrl(2),
         -- Master Port
         mAxisClk    => sysClk,
         mAxisRst    => sysRst,
         mAxisMaster => rxUartMaster,
         mAxisSlave  => rxUartSlave);

end mapping;
