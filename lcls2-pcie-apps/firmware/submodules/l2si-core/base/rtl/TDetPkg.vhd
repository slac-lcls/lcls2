-------------------------------------------------------------------------------
-- Title      : TDetPkg
-------------------------------------------------------------------------------
-- File       : TDetPkg.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2018-07-20
-- Last update: 2018-10-30
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

use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.EventPkg.all;

package TDetPkg is

   type TDetTimingType is record
      id           :  slv(31 downto 0);
      partition    :  slv( 2 downto 0);
      enable       :  sl;
      aFull        :  sl;
   end record TDetTimingType;
   type TDetTimingArray is array(natural range<>) of TDetTimingType;
   
   constant TDET_TIMING_INIT_C : TDetTimingType := (
      id           => (others=>'0'),
      partition    => (others=>'0'),
      enable       => '0',
      aFull        => '0' );

   type TDetTrigType is record
     l0a     : sl;
     l0tag   : slv( 4 downto 0);
     valid   : sl;
   end record;
   type TDetTrigArray is array(natural range<>) of TDetTrigType;
   constant TDETTRIG_INIT_C : TDetTrigType := (
     l0a     => '0',
     l0tag   => (others=>'0'),
     valid   => '0' );
   
   type TDetEventType is record
     header  : EventHeaderType;
     isEvent : sl;
     valid   : sl;
   end record TDetEventType;
   type TDetEventArray is array(natural range<>) of TDetEventType;
   constant TDETEVENT_INIT_C : TDetEventType := (
     header  => EVENT_HEADER_INIT_C,
     isEvent => '0',
     valid   => '0' );

   type TDetStatusType is record
     partitionAddr   : slv(31 downto 0);
     cntL0           : slv(19 downto 0);
     cntL1A          : slv(19 downto 0);
     cntL1R          : slv(19 downto 0);
     cntWrFifo       : slv( 3 downto 0);
     cntRdFifo       : slv( 3 downto 0);
     msgDelay        : slv( 6 downto 0);
     cntOflow        : slv( 7 downto 0);
   end record;
   type TDetStatusArray is array(natural range<>) of TDetStatusType;

   constant TDETSTATUS_BITS_C : natural := 117;

   constant TDET_AXIS_CONFIG_C : AxiStreamConfigType := (
     TSTRB_EN_C    => false,
     TDATA_BYTES_C => 24,
     TDEST_BITS_C  => 1,
     TID_BITS_C    => 0,
     TKEEP_MODE_C  => TKEEP_NORMAL_C,
     TUSER_BITS_C  => 2,
     TUSER_MODE_C  => TUSER_NORMAL_C );
   
   function toSlv       (status : TDetStatusType) return slv;
   function toTDetStatus(vector : slv           ) return TDetStatusType;
   
end package TDetPkg;

package body TDetPkg is

   function toSlv(status : TDetStatusType) return slv
   is
      variable vector  : slv(TDETSTATUS_BITS_C-1 downto 0) := (others=>'0');
      variable i       : integer := 0;
   begin
      assignSlv(i, vector, status.partitionAddr);
      assignSlv(i, vector, status.cntL0);
      assignSlv(i, vector, status.cntL1A);
      assignSlv(i, vector, status.cntL1R);
      assignSlv(i, vector, status.cntWrFifo);
      assignSlv(i, vector, status.cntRdFifo);
      assignSlv(i, vector, status.msgDelay);
      assignSlv(i, vector, status.cntOflow);
      return vector;
   end function;
      
   function toTDetStatus (vector : slv) return TDetStatusType
   is
      variable status  : TDetStatusType;
      variable i       : integer := 0;
   begin
      assignRecord(i, vector, status.partitionAddr);
      assignRecord(i, vector, status.cntL0);
      assignRecord(i, vector, status.cntL1A);
      assignRecord(i, vector, status.cntL1R);
      assignRecord(i, vector, status.cntWrFifo);
      assignRecord(i, vector, status.cntRdFifo);
      assignRecord(i, vector, status.msgDelay);
      assignRecord(i, vector, status.cntOflow);
      return status;
   end function;
   
end package body TDetPkg;
