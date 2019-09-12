-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmClkAndRst.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-08
-- Last update: 2017-08-29
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
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

library unisim;
use unisim.vcomponents.all;

entity XpmClkAndRst is
   generic (
      TPD_G         : time    := 1 ns;
      SIM_SPEEDUP_G : boolean := false);
   port (
      -- Reference Clocks and Resets
      ref125MHzClk : out sl;
      ref125MHzRst : out sl;
      ref156MHzClk : out sl;
      ref156MHzRst : out sl;
      ref312MHzClk : out sl;
      ref312MHzRst : out sl;
      ref625MHzClk : out sl;
      ref625MHzRst : out sl;
      gthFabClk    : out sl;
      -- AXI-Lite Clocks and Resets
      axilClk      : out sl;
      axilRst      : out sl;
      ----------------
      -- Core Ports --
      ----------------   
      -- Common Fabricate Clock
      fabClkP      : in  sl;
      fabClkN      : in  sl);
end XpmClkAndRst;

architecture mapping of XpmClkAndRst is

   signal gtClk     : sl;
   signal fabClk    : sl;
   signal fabRst    : sl;
   signal clk       : sl;
   signal rst       : sl;
   signal clkOut    : slv(2 downto 0);
   signal rstOut    : slv(2 downto 0);
   signal rstDly    : slv(2 downto 0);
   signal rstFO     : sl;
   
   --attribute dont_touch           : string;
   --attribute dont_touch of clk    : signal is "TRUE";
   --attribute dont_touch of rst    : signal is "TRUE";
   --attribute dont_touch of clkOut : signal is "TRUE";
   --attribute dont_touch of rstOut : signal is "TRUE";
   --attribute dont_touch of rstDly : signal is "TRUE";

begin

   ref156MHzClk <= fabClk;
   ref156MHzRst <= fabRst;

--   axilRst      <= rstDly(2);
--   ref156MHzRst <= rstDly(2);
   --  Put large fanout reset onto BUFG
   axilClk      <= clkOut(2);
   axilRst      <= rstFO;
   U_AXILRST : BUFG
     port map ( O => rstFO,
                I => rstDly(2) );

   -- Adding registers to help with timing
   process(clkOut)
   begin
      if rising_edge(clkOut(2)) then
         rstDly <= rstDly(1 downto 0) & rstOut(2) after TPD_G;
      end if;
   end process;

   ref125MHzClk <= clkOut(2);
   ref125MHzRst <= rstOut(2);

   ref312MHzClk <= clkOut(1);
   ref312MHzRst <= rstOut(1);

   -- Adding registers to help with timing
   process(clkOut)
   begin
      if rising_edge(clkOut(0)) then
         ref625MHzRst <= rstOut(0) after TPD_G;
      end if;
   end process;
   ref625MHzClk <= clkOut(0);

   IBUFDS_GTE3_Inst : IBUFDS_GTE3
      generic map (
         REFCLK_EN_TX_PATH  => '0',
         REFCLK_HROW_CK_SEL => "00",    -- 2'b00: ODIV2 = O
         REFCLK_ICNTL_RX    => "00")
      port map (
         I     => fabClkP,
         IB    => fabClkN,
         CEB   => '0',
         ODIV2 => gtClk,
         O     => gthFabClk);  

   BUFG_GT_Inst : BUFG_GT
      port map (
         I       => gtClk,
         CE      => '1',
         CEMASK  => '1',
         CLR     => '0',
         CLRMASK => '1',
         DIV     => "000",              -- Divide by 1
         O       => fabClk);

   PwrUpRst_Inst : entity work.PwrUpRst
      generic map(
         TPD_G         => TPD_G,
         SIM_SPEEDUP_G => SIM_SPEEDUP_G)
      port map(
         clk    => fabClk,
         rstOut => fabRst); 

   clk <= fabClk;
   rst <= fabRst;

   U_ClkManagerMps : entity work.ClockManagerUltraScale
      generic map(
         TPD_G              => TPD_G,
         TYPE_G             => "MMCM",
         INPUT_BUFG_G       => false,
         FB_BUFG_G          => true,
         RST_IN_POLARITY_G  => '1',
         NUM_CLOCKS_G       => 3,
         -- MMCM attributes
         BANDWIDTH_G        => "OPTIMIZED",
         CLKIN_PERIOD_G     => 6.4,
         DIVCLK_DIVIDE_G    => 1,
         CLKFBOUT_MULT_F_G  => 8.0,  -- 1.25 GHz
         CLKOUT0_DIVIDE_F_G => 2.0,                         -- 625 MHz = 1.25 GHz/2.0
         CLKOUT1_DIVIDE_G   => 4,                           -- 312.5 MHz = 1.25 GHz/4
         CLKOUT2_DIVIDE_G   => 10)                          -- 125 MHz = 1.25 GHz/10
      port map(
         -- Clock Input
         clkIn  => clk,
         rstIn  => rst,
         -- Clock Outputs
         clkOut => clkOut,
         -- Reset Outputs
         rstOut => rstOut);

end mapping;
