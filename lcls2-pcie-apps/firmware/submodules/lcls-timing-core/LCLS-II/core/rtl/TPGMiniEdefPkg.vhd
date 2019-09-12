-------------------------------------------------------------------------------
-- Title      : TPGMiniEdefPkg
-------------------------------------------------------------------------------
-- File       : TPGMiniEdefPkg.vhd
-- Author     : Till Straumann <strauman@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2018-03-08
-- Last update: 2018-03-08
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

package TPGMiniEdefPkg is

   subtype EdefNavgType is slv( 7 downto 0);
   subtype EdefNsmpType is slv(11 downto 0);
   subtype EdefType     is slv( 3 downto 0);
   subtype EdefTSType   is slv( 2 downto 0);
   subtype EdefRateType is slv( 2 downto 0);
   subtype EdefSevrType is slv( 1 downto 0);

   constant EDEF_NUM_BITS_C : natural :=   EdefNavgType'length
                                         + EdefNsmpType'length
                                         + EdefType'length
                                         + EdefTSType'length
                                         + EdefRateType'length
                                         + EdefSevrType'length;

   -- EDEF Parameters
   type TPGMiniEdefConfigType is record
      -- number of input values to average into one output sample
      navg : EdefNavgType;
      -- number of output samples to acquire (each of which is an
      -- average over 'navg' input values)
      nsmp : EdefNsmpType;
      -- EDEF 'instance' to control.
      edef : EdefType;
      -- Time-slot (0..5 corresponding to TS1..TS6); if the 'rate'
      -- is 120Hz then BSA is active during both timeslots; e.g.,
      -- rate == 0, slot == 0 is equivalent to rate == 0, slot == 3.
      slot : EdefTSType;
      -- BSA Rate (one of the base rates 0..6, i.e.,
      -- 120, 60, 30, 10, 5, 1 or 0.5Hz).
      rate : EdefRateType;
      -- BSA severity level ("00": reject only invalid;
      --                     "10": reject invalid, major;
      --                     "01": reject invalid, major, minor).
      sevr : EdefSevrType;
      -- write-enable (qualifies parameters)
      wrEn : sl;
   end record TPGMiniEdefConfigType;

   constant TPG_MINI_EDEF_CONFIG_INIT_C : TPGMiniEdefConfigType := (
      navg => (others => '0'),
      nsmp => (others => '0'),
      edef => (others => '0'),
      slot => (others => '0'),
      rate => (others => '0'),
      sevr => (others => '0'),
      wrEn => '0'
   );

   function toSlv(cfg : TPGMiniEdefConfigType) return slv;
   function fromSlv(v : slv; wen : sl        ) return TPGMiniEdefConfigType;

end package TPGMiniEdefPkg;

package body TPGMiniEdefPkg is

   function toSlv(cfg : TPGMiniEdefConfigType) return slv is
      variable i : integer;
      variable v : slv(EDEF_NUM_BITS_C - 1 downto 0);
   begin
      i := 0;
      assignSlv( i, v, cfg.nsmp );
      assignSlv( i, v, cfg.rate );
      assignSlv( i, v, cfg.slot );
      assignSlv( i, v, cfg.sevr );
      assignSlv( i, v, cfg.edef );
      assignSlv( i, v, cfg.navg );
      return v;
   end function toSlv;

   function fromSlv(v : slv; wen : sl) return TPGMiniEdefConfigType is
      variable i   : integer;
      variable cfg : TPGMiniEdefConfigType;
   begin
      i := 0;
      assignRecord( i, v, cfg.nsmp );
      assignRecord( i, v, cfg.rate );
      assignRecord( i, v, cfg.slot );
      assignRecord( i, v, cfg.sevr );
      assignRecord( i, v, cfg.edef );
      assignRecord( i, v, cfg.navg );
      cfg.wrEn := wen;
      return cfg;
   end function fromSlv;

end package body TPGMiniEdefPkg;
