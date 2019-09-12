------------------------------------------------------------------------------
-- This file is part of 'LCLS2 DAQ Software'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 DAQ Software', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;
use work.StdRtlPkg.all;

library unisim;
use unisim.vcomponents.all;

entity XpmPllClk is
  port ( clkIn       : in  sl;
         rstIn       : in  sl;
         clkOutP     : out slv(3 downto 0);
         clkOutN     : out slv(3 downto 0) );
end XpmPllClk;

architecture rtl of XpmPllClk is

  signal clkDdr : sl;
  signal clk, rst : sl;
  
begin

  --
  --  Can't use the ODDRE1 at fpgaclk_P/N(2) because it shares the
  --  BITSLICE with backplane SALT channel 4
  --
  --U_FPGACLK0 : entity work.ClkOutBufDiff
  --  generic map (
  --    XIL_DEVICE_G => "ULTRASCALE")
  --  port map (
  --    clkIn   => recTimingClk,
  --    clkOutP => fpgaclk_P(0),
  --    clkOutN => fpgaclk_N(0));

  --U_FPGACLK2 : entity work.ClkOutBufDiff
  --  generic map (
  --    XIL_DEVICE_G => "ULTRASCALE")
  --  port map (
  --    clkIn   => recTimingClk,
  --    clkOutP => fpgaclk_P(2),
  --    clkOutN => fpgaclk_N(2));

  U_MMCM : entity work.ClockManagerUltrascale
    generic map ( INPUT_BUFG_G     => false,
                  NUM_CLOCKS_G     => 1,
                  CLKIN_PERIOD_G   => 5.4,
                  CLKFBOUT_MULT_G  => 6,
                  CLKOUT0_DIVIDE_G => 3 )
    port map    ( clkIn     => clkIn,
                  rstIn     => rstIn,
                  clkOut(0) => clk,
                  rstOut(0) => rst );

   seq: process (clk)
   begin
     if rising_edge(clk) then
       if rst = '1' then
         clkDdr <= '0';
       else
         clkDdr <= not clkDdr;
       end if;
     end if;
   end process seq;
  
     -- Differential output buffer
   U_OBUF_0 : OBUFTDS
      port map (
         I  => clkDdr,
         T  => '0',
         O  => clkOutP(0),
         OB => clkOutN(0));

  U_OBUF_2 : OBUFTDS
      port map (
         I  => clkDdr,
         T  => '0',
         O  => clkOutP(2),
         OB => clkOutN(2));

  clkOutP(1) <= '0';
  clkOutN(1) <= '1';

  clkOutP(3) <= '0';
  clkOutN(3) <= '1';

end rtl;
