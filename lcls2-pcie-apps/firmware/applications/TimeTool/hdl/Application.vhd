-------------------------------------------------------------------------------
-- File       : Application.vhd
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
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.AppPkg.all;

entity Application is
   generic (
      TPD_G           : time             := 1 ns;
      AXI_BASE_ADDR_G : slv(31 downto 0) := x"00C0_0000");
   port (
      -- AXI-Lite Interface
      axilClk         : in  sl;
      axilRst         : in  sl;
      axilReadMaster  : in  AxiLiteReadMasterType;
      axilReadSlave   : out AxiLiteReadSlaveType;
      axilWriteMaster : in  AxiLiteWriteMasterType;
      axilWriteSlave  : out AxiLiteWriteSlaveType;
      -- PGP Streams (axilClk domain)
      pgpIbMasters    : out AxiStreamMasterArray(3 downto 0)    := (others => AXI_STREAM_MASTER_INIT_C);
      pgpIbSlaves     : in  AxiStreamSlaveArray(3 downto 0);
      pgpObMasters    : in  AxiStreamQuadMasterArray(3 downto 0);
      pgpObSlaves     : out AxiStreamQuadSlaveArray(3 downto 0) := (others => (others => AXI_STREAM_SLAVE_FORCE_C));
      -- Trigger Event streams (axilClk domain)
      trigMasters     : in  AxiStreamMasterArray(3 downto 0);
      trigSlaves      : out AxiStreamSlaveArray(3 downto 0)     := (others => AXI_STREAM_SLAVE_FORCE_C);
      -- DMA Interface (dmaClk domain)
      dmaClk          : in  sl;
      dmaRst          : in  sl;
      dmaIbMasters    : out AxiStreamMasterArray(DMA_SIZE_C-1 downto 0);
      dmaIbSlaves     : in  AxiStreamSlaveArray(DMA_SIZE_C-1 downto 0);
      dmaObMasters    : in  AxiStreamMasterArray(DMA_SIZE_C-1 downto 0);
      dmaObSlaves     : out AxiStreamSlaveArray(DMA_SIZE_C-1 downto 0));
end Application;

architecture mapping of Application is

   constant AXIL_CONFIG_C : AxiLiteCrossbarMasterConfigArray(3 downto 0) := genAxiLiteConfig(4, AXI_BASE_ADDR_G, 22, 20);

   signal axilWriteMasters : AxiLiteWriteMasterArray(3 downto 0);
   signal axilWriteSlaves  : AxiLiteWriteSlaveArray(3 downto 0) := (others => AXI_LITE_WRITE_SLAVE_EMPTY_OK_C);
   signal axilReadMasters  : AxiLiteReadMasterArray(3 downto 0);
   signal axilReadSlaves   : AxiLiteReadSlaveArray(3 downto 0)  := (others => AXI_LITE_READ_SLAVE_EMPTY_OK_C);

begin

   --------------------
   -- AXI-Lite Crossbar
   --------------------
   U_AXIL_XBAR : entity work.AxiLiteCrossbar
      generic map (
         TPD_G              => TPD_G,
         NUM_SLAVE_SLOTS_G  => 1,
         NUM_MASTER_SLOTS_G => 4,
         MASTERS_CONFIG_G   => AXIL_CONFIG_C)
      port map (
         axiClk              => axilClk,
         axiClkRst           => axilRst,
         sAxiWriteMasters(0) => axilWriteMaster,
         sAxiWriteSlaves(0)  => axilWriteSlave,
         sAxiReadMasters(0)  => axilReadMaster,
         sAxiReadSlaves(0)   => axilReadSlave,
         mAxiWriteMasters    => axilWriteMasters,
         mAxiWriteSlaves     => axilWriteSlaves,
         mAxiReadMasters     => axilReadMasters,
         mAxiReadSlaves      => axilReadSlaves);

   -------------------
   -- Application Lane
   -------------------
   GEN_VEC :
   for i in DMA_SIZE_C-1 downto 0 generate
      U_Lane : entity work.AppLane
         generic map (
            TPD_G           => TPD_G,
            AXI_BASE_ADDR_G => AXIL_CONFIG_C(i).baseAddr)
         port map (
            -- AXI-Lite Interface (axilClk domain)
            axilClk         => axilClk,
            axilRst         => axilRst,
            axilReadMaster  => axilReadMasters(i),
            axilReadSlave   => axilReadSlaves(i),
            axilWriteMaster => axilWriteMasters(i),
            axilWriteSlave  => axilWriteSlaves(i),
            -- PGP Streams (axilClk domain)
            pgpIbMaster     => pgpIbMasters(i),
            pgpIbSlave      => pgpIbSlaves(i),
            pgpObMasters    => pgpObMasters(i),
            pgpObSlaves     => pgpObSlaves(i),
            -- Trigger Event streams (axilClk domain)
            trigMaster      => trigMasters(i),
            trigSlave       => trigSlaves(i),
            -- DMA Interface (dmaClk domain)
            dmaClk          => dmaClk,
            dmaRst          => dmaRst,
            dmaIbMaster     => dmaIbMasters(i),
            dmaIbSlave      => dmaIbSlaves(i),
            dmaObMaster     => dmaObMasters(i),
            dmaObSlave      => dmaObSlaves(i));
   end generate GEN_VEC;

end mapping;
