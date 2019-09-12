-------------------------------------------------------------------------------
-- Title      : ClockTime_186MHz
-------------------------------------------------------------------------------
-- File       : ClockTime_186MHz.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-15
-- Last update: 2015/09/15
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Increments a 64-bit nanosecond timestamp in 1300/7 MHz steps
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
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
use work.StdRtlPkg.all;

entity ClockTime is
   generic (
      TPD_G    : time    := 1 ns)
   port ( 
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
architecture ClockTime_186MHz of ClockTime is 

  constant remainder : slv( 4 downto 0)  := slv(conv_unsigned( 5,5));
  constant divisor   : slv( 4 downto 0)  := slv(conv_unsigned(13,5));
  constant step      : slv(31 downto 0) := slv(conv_unsigned( 5,32));
  constant one_sec   : slv(31 downto 0) := slv(conv_unsigned(1000000000,32));
  signal valid          : sl;
  signal dataSL, dataNL, dataNU : slv(31 downto 0);
  signal wrDataB, dataB : slv(wrData'range);
  signal remB , remN  : slv( 4 downto 0);
  
begin
  U_WrFifo : entity work.SynchronizerFifo
    generic map ( TPD_G=> TPD_G, DATA_WIDTH_G => 64 )
    port map ( rst    => rst,
               wr_clk => clkA,
               wr_en  => wrEnA,
               din    => wrData,
               rd_clk => clkB,
               rd_en  => wrEnB,
               valid  => valid,
               dout   => wrDataB );

  U_RdFifo : entity work.SynchronizerFifo
    generic map ( TPD_G=> TPD_G, DATA_WIDTH_G => 64 )
    port map ( rst    => rst,
               wr_clk => clkB,
               wr_en  => wrEnB,
               din    => dataB,
               rd_clk => clkA,
               rd_en  => '1',
               valid  => open,
               dout   => rdData );

  dataSL <= (wrDataB(31 downto 0))      when (valid='1' and wrEnB='1')  else
            (dataB(31 downto 0)+step)   when (remB+remainder < divisor) else
            (dataB(31 downto 0)+step+1);

  dataNL <= (dataSL) when (dataSL<one_sec) else
            (dataSL-one_sec);

  dataNU <= (wrDataB(63 downto 32))      when (valid='1' and wrEnB='1')  else
            (dataB(63 downto 32))        when (dataSL<one_sec) else
            (dataB(63 downto 32)+1);

  remN  <= (others=>'0')    when (valid='1' and wrEnB='1') else
           (remB+remainder) when (remB+remainder < divisor) else
           (remB+remainder-divisor);
  
  process (clkB, rst)
  begin
    if rst='1' then
      dataB <= (others=>'0');
    elsif rising_edge(clkB) then
      dataB <= dataNU & dataNL after TPD_G;
      remB  <= remN after TPD_G;
    end if;
  end process;

  dataO <= dataB;
  
end ClockTime_186MHz;
