-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmLinkClocks.vhd
-- Author     : Matt Weaver
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-12-14
-- Last update: 2016-05-26
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Distribution of MGT clocks
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 XPM Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 XPM Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

use work.StdRtlPkg.all;
use work.XpmPkg.all;

library unisim;
use unisim.vcomponents.all;

entity XpmLinkClocks is
   port (
     clkP             : in  sl;
     clkN             : in  sl;
     clkO             : out slv(NDSLinks-1 downto 0) );
end XpmLinkClocks;

architecture rtl of XpmLinkClocks is

  signal clkRef : sl;
  
begin
   XPM_IBUFDS_GTE3 : IBUFDS_GTE3
      generic map (
         REFCLK_EN_TX_PATH  => '0',
         REFCLK_HROW_CK_SEL => "01",    -- 2'b01: ODIV2 = Divide-by-2 version of O
         REFCLK_ICNTL_RX    => "00")
      port map (
         I     => clkP,
         IB    => clkN,
         CEB   => '0',
         ODIV2 => open,
         O     => clkRef);

   GEN_REFCLK: for i in 0 to NDSLinks-1 generate
     clkO(i) <= clkRef;
   end generate;
end rtl;
