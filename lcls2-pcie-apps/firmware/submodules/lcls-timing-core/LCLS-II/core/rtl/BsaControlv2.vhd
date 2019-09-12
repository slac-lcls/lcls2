-------------------------------------------------------------------------------
-- Title         : BsaControl
-- Project       : LCLS-II Timing Pattern Generator
-------------------------------------------------------------------------------
-- File          : BsaControl.vhd
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
use work.StdRtlPkg.all;

entity BsaControl is
  generic ( TPD_G    : time    := 1 ns; ASYNC_REGCLK_G : boolean := false );
  port (
      sysclk     : in  sl;
      sysrst     : in  sl;
      bsadef     : in  BsaDefType;
      tmocnt     : in  slv( 3 downto 0) := x"F";  -- 10 ms steps
      nToAvgOut  : out slv(15 downto 0);
      avgToWrOut : out slv(15 downto 0);

      txclk      : in  sl;
      txrst      : in  sl;
      enable     : in  sl;
      fixedRate  : in  slv(FIXEDRATEDEPTH-1 downto 0);
      acRate     : in  slv(ACRATEDEPTH-1 downto 0);
      acTS       : in  slv(2 downto 0);
      beamSeq    : in  slv(31 downto 0);
      expSeq     : in  Slv16Array(0 to 17);
      bsaInit    : out sl;
      bsaActive  : out sl;
      bsaAvgDone : out sl;
      bsaDone    : out sl
      );
end BsaControl;

architecture BsaControl of BsaControl is

   type RegType is record
     bsaInit    : sl;
     bsaActive  : sl;
     bsaAvgDone : sl;
     bsaDone    : sl;
     initd      : sl;
     initq      : sl;
     persist    : sl;
     done       : sl;
     lastAvg    : sl;
     lastWr     : sl;
     fifoRst    : sl;
     nToAvg     : slv(12 downto 0);
     avgToWr    : slv(15 downto 0);
     tmoactive  : sl;
     tmocnt     : slv( 3 downto 0);
   end record;

   constant REG_INIT_C : RegType := (
     bsaInit    => '0',
     bsaActive  => '0',
     bsaAvgDone => '0',
     bsaDone    => '0',
     initd      => '0',
     initq      => '0',
     persist    => '0',
     done       => '1',
     lastAvg    => '0',
     lastWr     => '0',
     fifoRst    => '0',
     nToAvg     => (others=>'0'),
     avgToWr    => (others=>'0'),
     tmoactive  => '0',
     tmocnt     => (others=>'0') );

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal rateSel : sl;
   
   -- Register delay for simulation
   constant tpd : time := 0.5 ns;

begin

   bsaInit    <= r.bsaInit;
   bsaActive  <= r.bsaActive;
   bsaAvgDone <= r.bsaAvgDone;
   bsaDone    <= r.bsaDone;
   
   U_Select : entity work.EventSelect
     generic map (TPD_G=>TPD_G)
     port map ( clk       => txclk,
                rateType  => bsadef.rateSel(12 downto 11),
                fxRateSel => bsadef.rateSel( 3 downto 0),
                acRateSel => bsadef.rateSel( 2 downto 0),
                acTSmask  => bsadef.rateSel( 8 downto 3),
                seqword   => bsadef.rateSel(10 downto 5),
                seqbit    => bsadef.rateSel( 4 downto 0),
                fixedRate => fixedRate,
                acRate    => acRate,
                acTS      => acTS,
                expSeq    => expSeq,
                rateSel   => rateSel );
                

   GEN_ASYNC: if ASYNC_REGCLK_G=true generate
     U_SynchFifo : entity work.SynchronizerFifo
       generic map (TPD_G=>TPD_G,
                    DATA_WIDTH_G => 32,
                    ADDR_WIDTH_G => 2)
       port map (rst                => rin.fifoRst,
                 wr_clk             => txclk,
                 wr_en              => '1',
                 din(12 downto 0)   => r.nToAvg,
                 din(15 downto 13)  => "000",
                 din(31 downto 16)  => r.avgToWr,
                 rd_clk             => sysclk,
                 rd_en              => '1',
                 valid              => open,
                 dout(15 downto 0)  => nToAvgOut,
                 dout(31 downto 16) => avgToWrOut);
   end generate GEN_ASYNC;

   GEN_SYNC: if ASYNC_REGCLK_G=false generate
     nToAvgOut  <= "000" & r.nToAvg;
     avgToWrOUt <= r.avgToWr;
   end generate GEN_SYNC;
   
   comb: process (r, txrst, enable, bsadef, beamSeq, rateSel, fixedRate, tmocnt) is
     variable v : RegType;
     variable destSel : sl;
     variable avgDone : sl;
   begin
     v := r;

     if r.nToAvg+1 = bsadef.nToAvg then
       v.lastAvg := '1';
     else
       v.lastAvg := '0';
     end if;

     if r.avgToWr+1 = bsadef.avgToWr then
       v.lastWr := '1';
     else
       v.lastWr := '0';
     end if;
     
     v.fifoRst := r.initq and not r.initd;
     
     if enable='1' then

       if ((bsadef.destSel(17 downto 16)="10") or
           (bsadef.destSel(17 downto 16)="01" and not (beamSeq(0)='1' and bsadef.destSel(conv_integer(beamSeq(7 downto 4))) = '1')) or
           (bsadef.destSel(17 downto 16)="00" and beamSeq(0)='1' and bsadef.destSel(conv_integer(beamSeq(7 downto 4))) = '1')) then
         destSel := '1';
       else
         destSel := '0';
       end if;

       v.initd     := r.initq;
       v.bsaInit   := r.initq and not r.initd;

       if v.bsaInit='1' then
         v.bsaActive := bsadef.maxSevr(0);
       else
         v.bsaActive := rateSel and destSel and not r.done;
       end if;

       if v.bsaInit='1' then
         v.bsaAvgDone := bsadef.maxSevr(1);
       else
         if (r.lastAvg='1' and v.bsaActive='1') then
           v.bsaAvgDone := '1';
         else
           v.bsaAvgDone := '0';
         end if;
       end if;
       
       if v.bsaInit='1' then
         v.done    := '0';
       else
         if (r.persist='0' and r.lastWr='1' and v.bsaAvgDone='1') then
           v.done  := '1';
         end if;
       end if;

       v.bsaDone := (v.done and not r.done);
       
       if (v.bsaInit='1' or v.bsaAvgDone='1') then
         v.nToAvg  := (others=>'0');
       elsif v.bsaActive='1' then
         v.nToAvg  := r.nToAvg+1;
       end if;

       if v.bsaInit='1' then
         v.avgToWr   := (others=>'0');
       elsif v.bsaAvgDone='1' then
         v.avgToWr   := r.avgToWr+1;
       end if;

       -- count off the tmo in 10 ms steps
       if r.tmoactive = '1' and fixedRate(4)='1' then
         if r.tmocnt = tmocnt then
           --  assert done
           v.bsaDone   := '1';
           v.tmoactive := '0';
           v.tmocnt    := (others=>'0');
         else
           v.tmocnt    := r.tmocnt+1;
         end if;
       end if;

       if r.bsaDone = '1' then
         v.tmoactive := '0';
       elsif r.bsaInit = '0' and r.bsaAvgDone = '1' then
         v.tmoactive := not bsadef.noTmo;
       end if;
       
     end if;

     if bsadef.init='0' then
       v.initq   := '0';
       v.persist := '0';
     elsif enable='1' and r.initq='0' then
       v.initq   := '1';
       if bsadef.avgToWr=x"FFFF" then
         v.persist := '1';
       end if;
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
   
end BsaControl;

