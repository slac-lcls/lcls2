-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : EventPkg.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2016-03-25
-- Last update: 2018-12-14
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Programmable configuration and status fields
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
use work.TimingPkg.all;

package EventPkg is

   type TimingHeaderType is record
      strobe    : sl;
      pulseId   : slv(63 downto 0);
      timeStamp : slv(63 downto 0);
   end record;
   constant TIMING_HEADER_INIT_C : TimingHeaderType := (
      strobe    => '0',
      pulseId   => (others=>'0'),
      timeStamp => (others=>'0') );

   function toTimingHeader(v : TimingBusType) return TimingHeaderType;

   
   constant EVENT_HEADER_VERSION_C : integer := 0;
   constant L1A_INFO_C : slv(6 downto 0) := toSlv(12,7);
   
   type EventHeaderType is record
     pulseId    : slv(63 downto 0);
     timeStamp  : slv(63 downto 0);
     count      : slv(23 downto 0);
     version    : slv( 7 downto 0);
     partitions : slv(15 downto 0);   -- readout groups
     l1t        : slv(15 downto 0);   -- L1 trigger lines
     payload    : slv( 7 downto 0);   -- transition payload
     damaged    : sl;
   end record;

   constant EVENT_HEADER_INIT_C : EventHeaderType := (
     pulseId    => (others=>'0'),
     timeStamp  => (others=>'0'),
     count      => (others=>'0'),
     version    => toSlv(EVENT_HEADER_VERSION_C,8),
     partitions => (others=>'0'),
     l1t        => (others=>'0'),
     payload    => (others=>'0'),
     damaged    => '0' );

   type EventHeaderArray is array(natural range<>) of EventHeaderType;
   
   function toSlv(v : EventHeaderType) return slv;
   
end package EventPkg;

package body EventPkg is

   function toTimingHeader(v : TimingBusType) return TimingHeaderType is
     variable result : TimingHeaderType;
   begin
     result.strobe    := v.strobe;
     result.pulseId   := v.message.pulseId;
     result.timeStamp := v.message.timeStamp;
     return result;
   end function;
   
   function toSlv(v : EventHeaderType) return slv is
     variable vector : slv(191 downto 0) := (others=>'0');
     variable i      : integer := 0;
   begin
     assignSlv(i, vector, v.pulseId(55 downto 0));
     if v.l1t(15) = '1' then
       assignSlv(i, vector, L1A_INFO_C);
     else
       assignSlv(i, vector, v.l1t(12 downto 6)); 
     end if;
     assignSlv(i, vector, v.damaged   );
     assignSlv(i, vector, v.timeStamp );
     assignSlv(i, vector, v.count     );
     assignSlv(i, vector, v.version   );
     if v.l1t(15) = '1' then
       assignSlv(i, vector, v.partitions);
     else
       assignSlv(i, vector, v.payload);
       assignSlv(i, vector, x"00");
     end if;
     assignSlv(i, vector, v.l1t       );
     return vector;
   end function;

   
end package body EventPkg;
