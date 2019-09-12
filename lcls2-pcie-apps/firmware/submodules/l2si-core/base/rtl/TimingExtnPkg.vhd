-------------------------------------------------------------------------------
-- Title      : TimingPkg
-------------------------------------------------------------------------------
-- File       : TimingExtnPkg.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2018-07-20
-- Last update: 2018-07-21
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

   constant EXPT_STREAM_ID    : slv(3 downto 0) := x"1";

   --
   --  Experiment timing information (appended by downstream masters)
   --
--   constant PADDR_LEN : integer := 16;
--   constant PWORD_LEN : integer := 32;
   constant PADDR_LEN : integer := 32;
   constant PWORD_LEN : integer := 48;
   constant EXPT_MESSAGE_BITS_C : integer := PADDR_LEN+8*PWORD_LEN;

   --type ExptMessageType is record
   --  partitionAddr   : slv(PADDR_LEN-1 downto 0);
   --  partitionWord   : Slv32Array(0 to 7);
   --end record;
   --constant EXPT_MESSAGE_INIT_C : ExptMessageType := (
   --  partitionAddr  => (others=>'1'),
   --  partitionWord  => (others=>x"80008000") );
   type ExptMessageType is record
     partitionAddr   : slv(31 downto 0);
     partitionWord   : Slv48Array(0 to 7);
   end record;
   constant EXPT_MESSAGE_INIT_C : ExptMessageType := (
     partitionAddr  => (others=>'1'),
     partitionWord  => (others=>x"800080008000") );
   type ExptMessageArray is array (integer range<>) of ExptMessageType;

   type ExptBusType is record
     message : ExptMessageType;
     valid   : sl;
   end record ExptBusType;
   constant EXPT_BUS_INIT_C : ExptBusType := (
     message => EXPT_MESSAGE_INIT_C,
     valid   => '0' );
   type ExptBusArray is array (integer range<>) of ExptBusType;

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
      assignSlv(i, vector, message.partitionAddr);
      for j in message.partitionWord'range loop
         assignSlv(i, vector, message.partitionWord(j));
      end loop;
      return vector;
   end function;
      
   function toExptMessageType (vector : slv) return ExptMessageType
   is
      variable message : ExptMessageType;
      variable i       : integer := 0;
   begin
      assignRecord(i, vector, message.partitionAddr);
      for j in message.partitionWord'range loop
         assignRecord(i, vector, message.partitionWord(j));
      end loop;
      return message;
   end function;
   
--   function toSlv(message : TimingExtnType) return slv is
--   begin
--     return toSlv(ExptMessageType(message));
--   end function;
   
--   function toTimingExtnType (vector : slv) return TimingExtnType is
--   begin
--     return TimingExtnType(toExptMessageType(vector));
--   end function;
   function toTimingExtnType (vector : slv) return TimingExtnType is
      variable message : TimingExtnType;
      variable i       : integer := 0;
   begin
      assignRecord(i, vector, message.partitionAddr);
      for j in message.partitionWord'range loop
         assignRecord(i, vector, message.partitionWord(j));
      end loop;
      return message;
   end function;   
   
end package body TimingExtnPkg;
