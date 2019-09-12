-------------------------------------------------------------------------------
-- Title      : ClockTime
-------------------------------------------------------------------------------
-- File       : ClockTime.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-15
-- Last update: 2016-07-12
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Increments a 64-bit nanosecond timestamp in programmable steps
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------
LIBRARY ieee;
use work.all;

USE ieee.std_logic_1164.ALL;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use work.StdRtlPkg.all;

entity ClockTime is
   generic (
      TPD_G    : time    := 1 ns;
      FRACTION_DEPTH_G : integer := 5
      );
   port (
      -- Defaults to a step duration of 5-5/13 nanoseconds (1300/7 MHz)
      step               : in slv( 4 downto 0) := slv(conv_unsigned( 5,5));
      remainder          : in slv( FRACTION_DEPTH_G-1 downto 0) := slv(conv_unsigned( 5,FRACTION_DEPTH_G));
      divisor            : in slv( FRACTION_DEPTH_G-1 downto 0) := slv(conv_unsigned(13,FRACTION_DEPTH_G));
      -- Clock and reset
      rst                : in  sl;
      clkA               : in  sl;
      wrEnA              : in  sl;
      wrData             : in  slv(63 downto 0);
      rdData             : out slv(63 downto 0);
      
      clkB               : in  sl;
      wrEnB              : in  sl;
      dataO              : out slv(63 downto 0)
      );
end ClockTime;

-- Define architecture for top level module
architecture rtl of ClockTime is 

  constant one_sec   : slv(31 downto 0) := slv(conv_unsigned(1000000000,32));
  signal step32         : slv (31 downto 0);
  signal valid          : sl;
  signal dataSL, dataNL, dataNU : slv(31 downto 0);
  signal wrDataB        : slv(wrData'range);
  signal dataB          : slv(wrData'range) := (others=>'0');
  signal remB           : slv( FRACTION_DEPTH_G downto 0) := (others=>'0');
  signal remN           : slv( FRACTION_DEPTH_G downto 0);
  signal urem           : slv( FRACTION_DEPTH_G downto 0);
  signal udiv           : slv( FRACTION_DEPTH_G downto 0);
begin
  step32 <= x"000000" & "000" & step;
  urem   <= '0' & remainder;
  udiv   <= '0' & divisor;
  
  U_WrFifo : entity work.SynchronizerFifo
    generic map ( TPD_G=>TPD_G, DATA_WIDTH_G => 64 )
    port map ( rst    => rst,
               wr_clk => clkA,
               wr_en  => wrEnA,
               din    => wrData,
               rd_clk => clkB,
               rd_en  => wrEnB,
               valid  => valid,
               dout   => wrDataB );

  U_RdFifo : entity work.SynchronizerFifo
    generic map ( TPD_G=>TPD_G, DATA_WIDTH_G => 64 )
    port map ( rst    => rst,
               wr_clk => clkB,
               wr_en  => wrEnB,
               din    => dataB,
               rd_clk => clkA,
               rd_en  => '1',
               valid  => open,
               dout   => rdData );

  dataSL <= (wrDataB(31 downto 0))      when (valid='1' and wrEnB='1')  else
            (dataB(31 downto 0)+step32) when (remB+urem < udiv) else
            (dataB(31 downto 0)+step32+1);

  dataNL <= (dataSL) when (dataSL<one_sec) else
            (dataSL-one_sec);

  dataNU <= (wrDataB(63 downto 32))      when (valid='1' and wrEnB='1')  else
            (dataB(63 downto 32))        when (dataSL<one_sec) else
            (dataB(63 downto 32)+1);

  remN  <= (others=>'0')    when (valid='1' and wrEnB='1') else
           (remB+urem) when (remB+urem < udiv) else
           (remB+urem-udiv);
  
  process (clkB, rst)
  begin
    if rst='1' then
      null;
    elsif rising_edge(clkB) then
      dataB <= dataNU & dataNL after TPD_G;
      remB  <= remN after TPD_G;
    end if;
  end process;

  dataO <= dataB;
  
end rtl;
