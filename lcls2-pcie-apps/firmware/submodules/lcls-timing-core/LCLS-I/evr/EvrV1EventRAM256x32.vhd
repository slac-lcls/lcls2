-------------------------------------------------------------------------------
-- File       : EvrV1EventRAM256x32.vhd
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-02-17
-- Last update: 2015-02-17
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

entity EvrV1EventRAM256x32 is
   generic (
      TPD_G : time := 1 ns);
   port (
      -- Port A     
      clka  : in  sl;
      ena   : in  sl;
      wea   : in  sl;
      addra : in  slv(7 downto 0);
      dina  : in  slv(31 downto 0);
      douta : out slv(31 downto 0);
      -- Port B
      clkb  : in  sl;
      enb   : in  sl;
      web   : in  sl;
      addrb : in  slv(7 downto 0);
      dinb  : in  slv(31 downto 0);
      doutb : out slv(31 downto 0));   
end EvrV1EventRAM256x32;

architecture mapping of EvrV1EventRAM256x32 is

begin
   
   TrueDualPortRam_Inst : entity work.TrueDualPortRam
      generic map (
         TPD_G        => TPD_G,
         MODE_G       => "read-first",
         DATA_WIDTH_G => 32,
         ADDR_WIDTH_G => 8)
      port map (
         -- Port A     
         clka  => clka,
         ena   => ena,
         wea   => wea,
         addra => addra,
         dina  => dina,
         douta => douta,
         -- Port B
         clkb  => clkb,
         enb   => enb,
         web   => web,
         addrb => addrb,
         dinb  => dinb,
         doutb => doutb);   

end mapping;
