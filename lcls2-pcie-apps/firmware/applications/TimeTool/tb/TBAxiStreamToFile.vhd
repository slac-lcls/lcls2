-------------------------------------------------------------------------------
-- File       : TBAxiStreamToFile.vhd
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2017-10-24
-- Last update: 2018-11-08
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'axi-pcie-dev'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'axi-pcie-dev', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

use work.StdRtlPkg.all;
use work.AxiPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.AxiPciePkg.all;
use work.TimingPkg.all;
use work.Pgp2bPkg.all;
use work.SsiPkg.all;
use work.TestingPkg.all;

use STD.textio.all;
use ieee.std_logic_textio.all;

entity TBAxiStreamToFile is end TBAxiStreamToFile;

architecture testbed of TBAxiStreamToFile is

   constant TEST_OUTPUT_FILE_NAME : string := TEST_FILE_PATH & "/output_results.dat";

   constant AXI_BASE_ADDR_G   : slv(31 downto 0) := x"00C0_0000";

   constant TPD_G             : time             := 1 ns;

   constant DMA_SIZE_C        : positive         := 1;
 
   ----------------------------
   ----------------------------
   ----------------------------


   constant DMA_AXIS_CONFIG_G           : AxiStreamConfigType := ssiAxiStreamConfig(16, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8, 2);
   constant DMA_AXIS_DOWNSIZED_CONFIG_G : AxiStreamConfigType := ssiAxiStreamConfig(1, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 1, 2);

   constant CLK_PERIOD_G : time := 10 ns;

   constant SRC_CONFIG_C : AxiStreamConfigType := (
      TSTRB_EN_C    => false,
      TDATA_BYTES_C => 4, -- 128 bits
      TDEST_BITS_C  => 0,
      TID_BITS_C    => 0,
      TKEEP_MODE_C  => TKEEP_COMP_C,
      TUSER_BITS_C  => 2,
      TUSER_MODE_C  => TUSER_FIRST_LAST_C);

   signal appInMaster  : AxiStreamMasterType;
   signal appInSlave   : AxiStreamSlaveType;

   signal axiClk   : sl;
   signal axiRst   : sl;

begin

   --------------------
   -- Clocks and Resets
   --------------------
   U_axilClk : entity work.ClkRst
      generic map (
         CLK_PERIOD_G      => CLK_PERIOD_G,
         RST_START_DELAY_G => 0 ns,
         RST_HOLD_TIME_G   => 1000 ns)
      port map (
         clkP => axiClk,
         rst  => axiRst);

   --------------------
   -- Test data
   --------------------  

      U_CamOutput : entity work.FileToAxiStream
         generic map (
            TPD_G              => TPD_G,
            BYTE_SIZE_C        => 2+1,
            DMA_AXIS_CONFIG_G  => DMA_AXIS_CONFIG_G,
            CLK_PERIOD_G       => 23 ns)
         port map (
            sysClk         => axiClk,
            sysRst         => axiRst,
            dataOutMaster  => appInMaster,
            dataOutSlave   => appInSlave);


      U_FileInput : entity work.AxiStreamToFile
         generic map (
            TPD_G              => TPD_G,
            BYTE_SIZE_C        => 2+1,
            DMA_AXIS_CONFIG_G  => DMA_AXIS_CONFIG_G,
            CLK_PERIOD_G       => 24 ns)
         port map (
            sysClk         => axiClk,
            sysRst         => axiRst,
            dataInMaster   => appInMaster,
            dataInSlave    => appInSlave);

end testbed;
