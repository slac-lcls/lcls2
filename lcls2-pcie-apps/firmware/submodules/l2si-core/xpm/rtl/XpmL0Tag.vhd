-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmL0Tag.vhd
-- Author     : Matt Weaver
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-12-14
-- Last update: 2018-03-12
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Level-0 timestamp cache
--
-- This module caches the event timestamp information (XpmAcceptFrameType) for each
-- accepted Level-0 trigger and generates an associated index 'push_tag' for
-- use in downstream link Level-1 trigger communications.  The event timestamp
-- information is later retrieved by the 'pop' signal and 'pop_tag'.
--
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
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.TimingPkg.all;
use work.XpmPkg.all;

entity XpmL0Tag is
   generic ( TAG_WIDTH_G : integer := 32 );
   port (
      clk              : in  sl;
      rst              : in  sl;
      config           : in  XpmL0TagConfigType;
      clear            : in  sl;
      timingBus        : in  TimingBusType;
      push             : in  sl;
      skip             : in  sl;
      push_tag         : out slv(TAG_WIDTH_G-1 downto 0);
      pop              : in  sl;
      pop_tag          : in  slv(7 downto 0);
      pop_frame        : out XpmAcceptFrameType );
end XpmL0Tag;

architecture rtl of XpmL0Tag is
   type RegType is record
      tag    : slv(TAG_WIDTH_G-1 downto 0);
   end record;
   constant REG_INIT_C : RegType := (
      tag    => (others=>'0'));

   signal r    : RegType := REG_INIT_C;
   signal rin  : RegType;

   signal uclear : sl;
begin
   push_tag  <= r.tag;
   pop_frame <= XPM_ACCEPT_FRAME_INIT_C;

   U_SYNC: entity work.SynchronizerVector
      generic map ( WIDTH_G  => 1 )
      port map ( clk                   => clk,
                 dataIn (0)            => clear,
                 dataOut(0)            => uclear );
   
   comb: process (r, push, skip, uclear) is
      variable v : RegType;
   begin
      v := r;
      if (push='1' or skip='1') then
         v.tag := r.tag+1;
      end if;

      if (uclear='1') then
         v := REG_INIT_C;
      end if;
      
      rin <= v;
   end process comb;
   seq: process (clk) is
   begin
      if rising_edge(clk) then
         r <= rin;
      end if;
   end process seq;
end rtl;
