-------------------------------------------------------------------------------
-- File       : EvrV1HeartBeat.vhd
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-02-18
-- Last update: 2015-10-27
-------------------------------------------------------------------------------
-- Description: EvrHeartBeat LED output
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
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;

entity EvrV1HeartBeat is
   generic (
      TPD_G : time := 1 ns);
   port (
      reset            : in  sl;
      uSecDividerReg   : in  slv(31 downto 0);
      eventCode        : in  slv(7 downto 0);
      eventClk         : in  sl;
      heartBeatTimeOut : out sl);
end entity EvrV1HeartBeat;

architecture rtl of EvrV1HeartBeat is

   type RegType is record
      heartBeatTimeOut : sl;
      cnt              : slv(31 downto 0);
   end record RegType;
   constant REG_INIT_C : RegType := (
      heartBeatTimeOut => '0',
      cnt              => (others => '0'));

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

begin

   comb : process (eventCode, r, reset, uSecDividerReg) is
      variable v : RegType;
   begin
      v := r;

      v.heartBeatTimeOut := '0';

      if eventCode = x"7A" then
         v.cnt := uSecDividerReg;
      else
         v.cnt := r.cnt - 1;
      end if;

      if r.cnt = 0 then
         v.heartBeatTimeOut := '1';
      end if;

      if (reset = '1') then
         v := REG_INIT_C;
      end if;

      rin <= v;

      heartBeatTimeOut <= r.heartBeatTimeOut;
      
   end process comb;

   seq : process (eventClk) is
   begin
      if (rising_edge(eventClk)) then
         r <= rin after TPD_G;
      end if;
   end process seq;

end architecture rtl;
