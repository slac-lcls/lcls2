-------------------------------------------------------------------------------
-- Title         : CtrControl
-- Project       : LCLS-II Timing Pattern Generator
-------------------------------------------------------------------------------
-- File          : CtrControl.vhd
-- Author        : Matt Weaver, weaver@slac.stanford.edu
-- Created       : 07/17/2015
-------------------------------------------------------------------------------
-- Description:
-- Translation of BSA DEF to control bits in timing pattern
-- Status updates:  nToAvgOut, avgToWrOut count up
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
-- 07/17/2015: created.
-------------------------------------------------------------------------------
library ieee;
use work.all;
use work.TPGPkg.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library UNISIM;
use UNISIM.VCOMPONENTS.all;
use work.StdRtlPkg.all;

entity CtrControl is
  generic ( TPD_G    : time    := 1 ns; ASYNC_REGCLK_G : boolean := false ); 
  port (
      sysclk     : in  sl;
      sysrst     : in  sl;
      ctrdef     : in  CtrDefType;
      ctrrst     : in  sl;

      txclk      : in  sl;
      txrst      : in  sl;
      enable     : in  sl;
      fixedRate  : in  slv(FIXEDRATEDEPTH-1 downto 0);
      acRate     : in  slv(ACRATEDEPTH-1 downto 0);
      acTS       : in  slv(2 downto 0);
      beamSeq    : in  slv(31 downto 0);
      expSeq     : in  Slv16Array(0 to 17);
      count      : out slv(31 downto 0) );
end CtrControl;

architecture CtrControl of CtrControl is

   type RegType is record
     count     : slv(31 downto 0);
     latch     : slv(31 downto 0);
   end record;

   constant REG_INIT_C : RegType := (
     count     => (others=>'0'),
     latch     => (others=>'0') );

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal rateSel : sl;
   signal ctrlatch : sl;
   
  attribute use_dsp48      : string;
  attribute use_dsp48 of r : signal is "yes";   
  
begin

   count <= r.latch;

   U_Latch : entity work.SynchronizerEdge
     port map ( clk        => txclk,
                dataIn     => ctrrst,
                risingEdge => ctrlatch);
   
   U_Select : entity work.EventSelect
     port map ( clk       => txclk,
                rateType  => ctrdef.rateSel(12 downto 11),
                fxRateSel => ctrdef.rateSel( 3 downto 0),
                acRateSel => ctrdef.rateSel( 2 downto 0),
                acTSmask  => ctrdef.rateSel( 8 downto 3),
                seqword   => ctrdef.rateSel(10 downto 5),
                seqbit    => ctrdef.rateSel( 4 downto 0),
                fixedRate => fixedRate,
                acRate    => acRate,
                acTS      => acTS,
                expSeq    => expSeq,
                rateSel   => rateSel );
                

   comb: process (r, ctrlatch, txrst, enable, ctrdef, beamSeq, rateSel) is
     variable v : RegType;
     variable destSel : sl;
   begin
     v := r;

     if enable='1' then

       if ((ctrdef.destSel(17 downto 16)="10") or
           (ctrdef.destSel(17 downto 16)="01" and not (beamSeq(0)='1' and ctrdef.destSel(conv_integer(beamSeq(7 downto 4))) = '1')) or
           (ctrdef.destSel(17 downto 16)="00" and beamSeq(0)='1' and ctrdef.destSel(conv_integer(beamSeq(7 downto 4))) = '1')) then
         destSel := '1';
       else
         destSel := '0';
       end if;

       if rateSel='1' and destSel='1' then
         v.count := r.count+1;
       end if;
       
     end if;

     if ctrlatch='1' then
       v.latch := r.count;
       v.count := (others=>'0');
     end if;
     
     if txrst='1' then
       v := REG_INIT_C;
     end if;
     
     rin <= v;
   end process;

   seq: process(txclk) is
   begin
     if rising_edge(txclk) then
       r <= rin after TPD_G;
     end if;
   end process;
   
end CtrControl;

