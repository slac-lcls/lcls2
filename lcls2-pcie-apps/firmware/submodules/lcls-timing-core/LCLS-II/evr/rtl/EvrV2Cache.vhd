-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : EvrV2Cache.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2016-01-04
-- Last update: 2016-01-24
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;

entity EvrV2Cache is
  generic ( TPD_G : time := 1 ns);
  port ( rsta    : in  sl;
         clka    : in  sl;
         wea     : in  sl;
         din     : in  slv(191 downto 0);
         rstb    : in  sl;
         clkb    : in  sl;
         enb     : in  sl;
         empty   : out sl;
         dout    : out slv(31 downto 0) );
end EvrV2Cache;

architecture mapping of EvrV2Cache is

  type RegType is record
    addr : slv(9 downto 0);
    sel  : slv(2 downto 0);
    done : sl;
  end record;

  constant REG_TYPE_INIT_C : RegType := (
    addr => (others=>'0'),
    sel  => (others=>'0'),
    done => '0' );

  signal ra, rb : RegType := REG_TYPE_INIT_C;
  signal rin_a, rin_b : RegType;
  signal weaq : sl;
  signal doutb : slv(191 downto 0);
  
begin  -- mapping

  dout  <= doutb(conv_integer(rb.sel)*32+31 downto conv_integer(rb.sel)*32);
  empty <= rb.done;
  
  weaq <= wea and not ra.done;
  
  U_RAM : entity work.SimpleDualPortRam
    generic map ( DATA_WIDTH_G => 192,
                  ADDR_WIDTH_G =>  10 )
    port map ( clka  => clka,
               wea   => wea,
               addra => ra.addr,
               dina  => din,
               clkb  => clkb,
               addrb => rb.addr,
               doutb => doutb );
      
  process (ra, wea, rsta)
    variable v : RegType;
  begin  -- process
    v := ra;

    if allBits(ra.addr,'1') then 
      v.done := '1';
    elsif wea='1' then
      v.addr := ra.addr+1;
    end if;

    if rsta='1' then
      v := REG_TYPE_INIT_C;
    end if;

    rin_a <= v;
  end process;

  process (rb, enb, rstb)
    variable v : RegType;
  begin  -- process
    v := rb;

    if allBits(rb.addr,'1') then 
      v.done := '1';
    elsif enb='1' then
      if rb.sel="101" then
        v.sel  := "000";
        v.addr := rb.addr+1;
      else
        v.sel  := rb.sel+1;
      end if;
    end if;

    if rstb='1' then
      v := REG_TYPE_INIT_C;
    end if;
    
    rin_b <= v;
  end process;

  process (clka)
  begin
    if rising_edge(clka) then
      ra <= rin_a;
    end if;
  end process;
  
  process (clkb)
  begin
    if rising_edge(clkb) then
      rb <= rin_b;
    end if;
  end process;
  
end mapping;
