-------------------------------------------------------------------------------
-- File       : PgpLaneRx.vhd
-- Company    : SLAC National Accelerator Laboratory
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
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.Pgp3Pkg.all;

entity PgpLaneRx is
   generic (
      TPD_G            : time    := 1 ns;
      ROGUE_SIM_EN_G   : boolean := false;
      APP_AXI_CONFIG_G : AxiStreamConfigType;
      PHY_AXI_CONFIG_G : AxiStreamConfigType);
   port (
      -- AXIS Interface (axisClk domain)
      axisClk      : in  sl;
      axisRst      : in  sl;
      mAxisMasters : out AxiStreamQuadMasterType;
      mAxisSlaves  : in  AxiStreamQuadSlaveType;
      -- PGP Interface (pgpClk domain)
      pgpClk       : in  sl;
      pgpRst       : in  sl;
      rxlinkReady  : in  sl;
      pgpRxMasters : in  AxiStreamMasterArray(3 downto 0);
      pgpRxCtrl    : out AxiStreamCtrlArray(3 downto 0);
      pgpRxSlaves  : out AxiStreamSlaveArray(3 downto 0));
end PgpLaneRx;

architecture mapping of PgpLaneRx is

   signal pgpMasters : AxiStreamMasterArray(3 downto 0);
   signal rxMasters  : AxiStreamMasterArray(3 downto 0);
   signal rxSlaves   : AxiStreamSlaveArray(3 downto 0);

begin

   BLOWOFF_FILTER : process (pgpRxMasters, rxlinkReady) is
      variable tmp : AxiStreamMasterArray(3 downto 0);
      variable i   : natural;
   begin
      tmp := pgpRxMasters;
      for i in 3 downto 0 loop
         if (rxlinkReady = '0') then
            tmp(i).tValid := '0';
         end if;
      end loop;
      pgpMasters <= tmp;
   end process;

   GEN_VEC :
   for i in 3 downto 0 generate

      BUFFER_FIFO : entity work.AxiStreamFifoV2
         generic map (
            -- General Configurations
            TPD_G               => TPD_G,
            SLAVE_READY_EN_G    => ROGUE_SIM_EN_G,
            -- FIFO configurations
            BRAM_EN_G           => true,
            GEN_SYNC_FIFO_G     => false,
            FIFO_ADDR_WIDTH_G   => 9,
            FIFO_FIXED_THRESH_G => true,
            FIFO_PAUSE_THRESH_G => 128,
            -- AXI Stream Port Configurations
            SLAVE_AXI_CONFIG_G  => PHY_AXI_CONFIG_G,
            MASTER_AXI_CONFIG_G => APP_AXI_CONFIG_G)
         port map (
            -- Slave Port
            sAxisClk    => pgpClk,
            sAxisRst    => pgpRst,
            sAxisMaster => pgpMasters(i),
            sAxisCtrl   => pgpRxCtrl(i),
            sAxisSlave  => pgpRxSlaves(i),
            -- Master Port
            mAxisClk    => axisClk,
            mAxisRst    => axisRst,
            mAxisMaster => rxMasters(i),
            mAxisSlave  => rxSlaves(i));

      BURST_RESIZE_FIFO : entity work.AxiStreamFifoV2
         generic map (
            -- General Configurations
            TPD_G               => TPD_G,
            SLAVE_READY_EN_G    => true,
            VALID_THOLD_G       => 128,  -- Hold until enough to burst into the interleaving MUX
            VALID_BURST_MODE_G  => true,
            -- FIFO configurations
            BRAM_EN_G           => true,
            GEN_SYNC_FIFO_G     => true,
            FIFO_ADDR_WIDTH_G   => 9,
            -- AXI Stream Port Configurations
            SLAVE_AXI_CONFIG_G  => APP_AXI_CONFIG_G,
            MASTER_AXI_CONFIG_G => APP_AXI_CONFIG_G)
         port map (
            -- Slave Port
            sAxisClk    => axisClk,
            sAxisRst    => axisRst,
            sAxisMaster => rxMasters(i),
            sAxisSlave  => rxSlaves(i),
            -- Master Port
            mAxisClk    => axisClk,
            mAxisRst    => axisRst,
            mAxisMaster => mAxisMasters(i),
            mAxisSlave  => mAxisSlaves(i));

   end generate GEN_VEC;

end mapping;
