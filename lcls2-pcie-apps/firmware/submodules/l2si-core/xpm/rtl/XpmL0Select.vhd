-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmL0Select.vhd
-- Author     : Matt Weaver
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-12-14
-- Last update: 2018-10-31
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Level-0 trigger select
-- 
-- Select events for sensor integration based upon timing frame information
-- and programmed selection parameters.
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
use work.TPGPkg.all;
use work.XpmPkg.all;

entity XpmL0Select is
   generic ( DEBUG_G : boolean := false );
   port (
      clk              : in  sl;
      rst              : in  sl;
      -- programmed event selection criteria
      config           : in  XpmL0SelectConfigType;
      -- current pulse timing information
      timingBus        : in  TimingBusType;
      -- state of the deadtime assertion
      inhibit          : in  sl;
      -- strobe cycle when decision needs to be made
      strobe           : in  sl;
      -- event selection decision
      accept           : out sl;
      rejecc           : out sl;
      -- monitoring statistics
      status           : out XpmL0SelectStatusType );
end XpmL0Select;

architecture rtl of XpmL0Select is
   type RegType is record
      strobeRdy: sl;
      accept   : sl;
      rejecc   : sl;
      seqWord  : slv(15 downto 0);
      status   : XpmL0SelectStatusType;
   end record;
   constant REG_INIT_C : RegType := (
      strobeRdy=> '0',
      accept   => '0',
      rejecc   => '0',
      seqWord  => (others=>'0'),
      status   => XPM_L0_SELECT_STATUS_INIT_C );

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal uconfig : XpmL0SelectConfigType;

begin

   accept <= r.accept;
   rejecc <= r.rejecc;
   status <= r.status;

   U_SYNC: entity work.SynchronizerVector
      generic map ( WIDTH_G  => 34 )
      port map ( clk                   => clk,
                 dataIn (15 downto  0) => config.rateSel,
                 dataIn (31 downto 16) => config.destSel,
                 dataIn (32)           => config.reset,
                 dataIn (33)           => config.enabled,
                 dataOut(15 downto  0) => uconfig.rateSel,
                 dataOut(31 downto 16) => uconfig.destSel,
                 dataOut(32)           => uconfig.reset,
                 dataOut(33)           => uconfig.enabled);
                 
   comb: process (r, inhibit, timingBus, uconfig, strobe) is
      variable v : RegType;
      variable m       : TimingMessageType;
      variable rateSel : sl;
      variable destSel : sl;
      variable controlI : integer;
   begin
      v := r;

      v.accept := '0';
      v.rejecc := '0';
      
      m := timingBus.message; -- shorthand
      
      controlI      := conv_integer(uconfig.rateSel(13 downto 8));
      if (controlI<MAXEXPSEQDEPTH) then
        v.seqWord := m.control(controlI);
      else
        v.seqWord := (others=>'0');
      end if;

      if (timingBus.strobe='1') then
         v.strobeRdy := '1';
      end if;
      
      if (strobe='1' and r.strobeRdy='1') then
         v.strobeRdy := '0';
         -- calculate rateSel
         case uconfig.rateSel(15 downto 14) is
           when "00" => rateSel := m.fixedRates(conv_integer(uconfig.rateSel(3 downto 0)));
           when "01" => if (uconfig.rateSel(conv_integer(m.acTimeSlot)+3-1)='0') then
                          rateSel := '0';
                        else
                          rateSel := m.acRates(conv_integer(uconfig.rateSel(2 downto 0)));
                        end if;
           when "10" => rateSel := r.seqWord(conv_integer(uconfig.rateSel(3 downto 0)));
           when others => rateSel := '0';
         end case;
         -- calculate destSel
         if (uconfig.destSel(15)='1' or
             uconfig.destSel(conv_integer(m.beamRequest(7 downto 4)))='1') then
           destSel := '1';
         else
           destSel := '0';
         end if;
         if uconfig.enabled='1' then
           v.status.enabled := r.status.enabled+1;
           if (inhibit='1') then
             v.status.inhibited := r.status.inhibited+1;
           end if;
           if (rateSel='1' and destSel='1') then
             v.status.num := r.status.num+1;
             if (inhibit='1') then
               v.rejecc := '1';
               v.status.numInh := r.status.numInh+1;
             else
               v.accept := '1';
               v.status.numAcc := r.status.numAcc+1;
             end if;
           end if;
         end if;
      end if;

      if (uconfig.reset='1') then
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
