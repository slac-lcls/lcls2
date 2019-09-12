-------------------------------------------------------------------------------
-- Title      : TimingExtnPkg
-------------------------------------------------------------------------------
-- File       : TimingExtnPkg.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2018-07-20
-- Last update: 2018-08-04
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

package TimingExtnPkg is

   constant EXPT_STREAM_ID    : slv(3 downto 0) := x"2";

   constant EXPT_MESSAGE_BITS_C : integer := 320;

   type ExptMessageType is record
     baseRateSince1Hz : slv( 31 downto 0);
     baseRateSinceTM  : slv( 31 downto 0);
     irigTimeCode     : slv(255 downto 0);
   end record;
   constant EXPT_MESSAGE_INIT_C : ExptMessageType := (
     baseRateSince1Hz => (others=>'1'),
     baseRateSinceTM  => (others=>'1'),
     irigTimeCode     => (others=>'1') );

   function toSlv(message : ExptMessageType) return slv;
   function toExptMessageType (vector : slv) return ExptMessageType;

   -- The extended interface
   subtype TimingExtnType is ExptMessageType;
   constant TIMING_EXTN_INIT_C : ExptMessageType := EXPT_MESSAGE_INIT_C;
   constant TIMING_EXTN_BITS_C : integer := EXPT_MESSAGE_BITS_C;
--   function toSlv(message : TimingExtnType) return slv;
   function toTimingExtnType (vector : slv) return TimingExtnType;
   
end package TimingExtnPkg;

package body TimingExtnPkg is

   function toSlv(message : ExptMessageType) return slv
   is
      variable vector  : slv(EXPT_MESSAGE_BITS_C-1 downto 0) := (others=>'0');
      variable i       : integer := 0;
   begin
      assignSlv(i, vector, message.baseRateSince1Hz);
      assignSlv(i, vector, message.baseRateSinceTM);
      assignSlv(i, vector, message.irigTimeCode);
      return vector;
   end function;
      
   function toExptMessageType (vector : slv) return ExptMessageType
   is
      variable message : ExptMessageType;
      variable i       : integer := 0;
   begin
      assignRecord(i, vector, message.baseRateSince1Hz);
      assignRecord(i, vector, message.baseRateSinceTM);
      assignRecord(i, vector, message.irigTimeCode);
      return message;
   end function;
   
--   function toSlv(message : TimingExtnType) return slv is
--   begin
--     return toSlv(ExptMessageType(message));
--   end function;
   
   function toTimingExtnType (vector : slv) return TimingExtnType is
      variable message : TimingExtnType;
      variable i       : integer := 0;
   begin
      assignRecord(i, vector, message.baseRateSince1Hz);
      assignRecord(i, vector, message.baseRateSinceTM);
      assignRecord(i, vector, message.irigTimeCode);
      return message;
   end function;
   
end package body TimingExtnPkg;
