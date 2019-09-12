-------------------------------------------------------------------------------
-- File       : EvrV1TimeStampFIFO.vhd
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-02-17
-- Last update: 2015-10-27
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'LCLS1 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS1 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

use work.StdRtlPkg.all;

entity EvrV1TimeStampFIFO is
   generic (
      TPD_G : time := 1 ns);
   port (
      -- Asynchronous Reset
      rst           : in  sl;
      -- Write Ports (wr_clk domain)
      wr_clk        : in  sl;
      wr_en         : in  sl;
      din           : in  slv(71 downto 0);
      wr_data_count : out slv(8 downto 0);
      full          : out sl;
      -- Read Ports (rd_clk domain)
      rd_clk        : in  sl;
      rd_en         : in  sl;
      dout          : out slv(71 downto 0);
      rd_data_count : out slv(8 downto 0);
      empty         : out sl);
end EvrV1TimeStampFIFO;

architecture mapping of EvrV1TimeStampFIFO is

begin
   
   FifoAsync_Inst : entity work.FifoAsync
      generic map (
         TPD_G          => TPD_G,
         RST_POLARITY_G => '1',         -- '1' for active high rst, '0' for active low
         BRAM_EN_G      => true,
         FWFT_EN_G      => true,
         DATA_WIDTH_G   => 72,
         ADDR_WIDTH_G   => 9)
      port map (
         -- Asynchronous Reset
         rst           => rst,
         -- Write Ports (wr_clk domain)
         wr_clk        => wr_clk,
         wr_en         => wr_en,
         din           => din,
         wr_data_count => wr_data_count,
         overflow      => full,
         -- Read Ports (rd_clk domain)
         rd_clk        => rd_clk,
         rd_en         => rd_en,
         dout          => dout,
         rd_data_count => rd_data_count,
         empty         => empty);

end mapping;
