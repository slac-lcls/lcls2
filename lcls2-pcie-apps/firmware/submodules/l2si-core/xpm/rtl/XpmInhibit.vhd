-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmInhibit.vhd
-- Author     : Matt Weaver
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-12-14
-- Last update: 2017-09-25
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Level-0 trigger inhibit aggregation
--
-- Assert 'inhibit' as logical OR of link 'full' status for all enabled
-- links ('config').
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

use work.StdRtlPkg.all;
use work.XpmPkg.all;

entity XpmInhibit is
   port (
      -- register clock domain
      regclk           : in  sl;
      update           : in  sl;
      clear            : in  sl;                    -- clear statistics
      config           : in  XpmPartInhConfigType;  -- programmable parameters
      status           : out XpmInhibitStatusType;  -- statistics
      --  timing clock domain
      clk              : in  sl;
      rst              : in  sl;
      full             : in  slv(27 downto 0);      -- status from downstream links
      fiducial         : in  sl;
      l0Accept         : in  sl;
      l1Accept         : in  sl;
      rejecc           : in  sl;                    -- L0 rejected due to inhibit
      inhibit          : out sl );                  -- trigger inhibit status
end XpmInhibit;

architecture rtl of XpmInhibit is

   type RegType is record
     status  : XpmInhibitStatusType;
   end record;
   constant REG_INIT_C : RegType := (
     status => XPM_INHIBIT_STATUS_INIT_C );
   
   signal r    : RegType := REG_INIT_C;
   signal r_in : RegType;

   signal proginhb : slv(config.setup'range);
   signal fullb    : slv(27 downto 0);
   signal inhSrc   : slv(31 downto 0);
   signal counts   : SlVectorArray(31 downto 0, 31 downto 0);
   
begin
   status  <= r.status;
   inhibit <= uOr(fullb) or uOr(proginhb);

   inhSrc  <= (proginhb & fullb) when rejecc='1' else
              (others=>'0');

   U_SyncFull : entity work.SynchronizerVector
     generic map ( WIDTH_G => 28 )
     port map ( clk     => clk,
                dataIn  => full,
                dataOut => fullb );
       
   U_Status : entity work.SyncStatusVector
     generic map ( WIDTH_G => 32 )
     port map ( statusIn     => inhSrc,
                cntRstIn     => clear,
                rollOverEnIn => (others=>'1'),
                cntOut       => counts,
                wrClk        => clk,
                rdClk        => regclk );

   GEN_L0INH : for i in config.setup'range generate
     U_L0INH : entity work.XpmTrigInhibit
       port map ( rst        => rst,
                  clk        => clk,
                  config     => config.setup(i),
                  fiducial   => fiducial,
                  trig       => l0Accept,
                  inhibit    => proginhb    (i) );
   end generate;

   process (r, clear, counts, update) is
     variable v : RegType;
   begin
     v := r;

     if update='1' then
       for i in 0 to 31 loop
         v.status.counts(i) := muxSlVectorArray(counts,i);
       end loop;
     end if;

     if clear='1' then
       v := REG_INIT_C;
     end if;
     
     r_in <= v;
   end process;
   
   process (regclk) is
   begin
     if rising_edge(regclk) then
       r <= r_in;
     end if;
   end process;

end rtl;
