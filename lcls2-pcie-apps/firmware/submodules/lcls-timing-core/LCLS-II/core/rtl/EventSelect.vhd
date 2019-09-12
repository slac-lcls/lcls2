-------------------------------------------------------------------------------
-- Title         : EventSelect
-- Project       : LCLS-II Timing Pattern Generator
-------------------------------------------------------------------------------
-- File          : EventSelect.vhd
-- Author        : Matt Weaver, weaver@slac.stanford.edu
-- Created       : 03/07/2016
-------------------------------------------------------------------------------
-- Description:
-- Translation of BSA DEF to control bits in timing pattern
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------
-- Modification history:
-- 03/07/2016: created.
-------------------------------------------------------------------------------
library ieee;
use work.all;
use work.TPGPkg.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use work.StdRtlPkg.all;

entity EventSelect is
   generic (
      TPD_G    : time    := 1 ns);
  port (
    clk        : in  sl;
    -- criteria
    rateType   : in  slv(1 downto 0);
    fxRateSel  : in  slv(3 downto 0);
    acRateSel  : in  slv(2 downto 0);
    acTSmask   : in  slv(5 downto 0);
    seqword    : in  slv(5 downto 0);
    seqbit     : in  slv(4 downto 0);
    -- event
    fixedRate  : in  slv(FIXEDRATEDEPTH-1 downto 0);
    acRate     : in  slv(ACRATEDEPTH-1 downto 0);
    acTS       : in  slv(2 downto 0);
    expSeq     : in  Slv16Array(0 to 17);
    -- result
    rateSel    : out sl
    );
end EventSelect;

architecture rtl of EventSelect is
  signal expSeqWord       : slv(15 downto 0) := (others=>'0');
begin

  process (clk)
    variable expI : integer;
  begin
    if rising_edge(clk) then
      expI := conv_integer(seqword);
      if expI<18 then
        expSeqWord <= expSeq(expI) after TPD_G;
      else
        expSeqWord <= (others=>'0') after TPD_G;
      end if;
    end if;
  end process;

  process (rateType, fxRateSel, acRateSel, acTSmask,
           fixedRate, acTS, acRate, expSeqWord, seqbit)
  begin 
    case rateType is
      when "00" => rateSel <= fixedRate(conv_integer(fxRateSel));
      when "01" =>
        if (acTSmask(conv_integer(acTS)-1) = '0') then
          -- acTS counts from "1"
          rateSel <= '0';
        else
          rateSel <= acRate(conv_integer(acRateSel));
        end if;
      when "10"   => rateSel <= expSeqWord(conv_integer(seqbit));
      when others => rateSel <= '0';
    end case;
  end process;

end rtl;
