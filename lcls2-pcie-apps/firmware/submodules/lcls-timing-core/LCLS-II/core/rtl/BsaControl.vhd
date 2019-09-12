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

entity BsaControl is
  generic ( TPD_G    : time    := 1 ns; ASYNC_REGCLK_G : boolean := false ); 
  port (
      sysclk     : in  sl;
      sysrst     : in  sl;
      bsadef     : in  BsaDefType;
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

   signal initq, initd                      : sl := '0';
   signal initn                             : sl;
   signal done                              : sl := '1';
   signal donen                             : sl;
   signal persist                           : sl := '0';
   signal restart                           : sl := '0';
   signal active, rateSel, destSel, avgDone : sl;
   signal nToAvg                            : slv(14 downto 0) := (others=>'0');
   signal nToAvgn                           : slv(14 downto 0);
   signal avgToWr                           : slv(15 downto 0) := (others=>'0');
   signal avgToWrn                          : slv(15 downto 0);
   signal fifoRst                           : sl;
   signal control0, control1                : slv(35 downto 0);

   -- Register delay for simulation
   constant tpd : time := 0.5 ns;

begin

   U_Select : entity work.EventSelect
     generic map ( TPD_G=>TPD_G)
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
                
   process (txclk)
   begin
      if rising_edge(txclk) then
         if bsadef.init = '0' then
           initq      <= '0' after TPD_G;
           persist    <= '0' after TPD_G;
           restart    <= '0' after TPD_G;
         elsif enable = '1' and initq = '0' then
            initq <= '1'  after TPD_G;
            if bsadef.avgToWr = x"0000" then
               persist <= '1' after TPD_G;
            end if;
            restart <= bsadef.restart after TPD_G;
         end if;
      end if;
   end process;

   initn <= initq and not initd;

   destSel <= '1' when ((bsadef.destSel(17 downto 16)="10") or
                        (bsadef.destSel(17 downto 16)="01" and not (beamSeq(0)='1' and bsadef.destSel(conv_integer(beamSeq(7 downto 4))) = '1')) or
                        (bsadef.destSel(17 downto 16)="00" and beamSeq(0)='1' and bsadef.destSel(conv_integer(beamSeq(7 downto 4))) = '1')) else
              '0';
   active <= rateSel and destSel and ((restart and initd) or not done);
   donen  <= '0' when (initn = '1') else
             '1' when (persist = '0' and avgToWr = x"0001" and avgDone = '1') else
             '0' when (restart = '1') else
             done;
   avgDone <= '1' when (nToAvg = x"0001" and active = '1') else
              '0';
   avgToWrn <= bsadef.avgToWr when (initn = '1' or donen = '1' ) else
               avgToWr-1 when (avgDone = '1') else
               avgToWr;
   nToAvgn <= bsadef.nToAvg when (initn = '1' or avgDone = '1') else
              nToAvg-1 when (active = '1') else
              nToAvg;
   fifoRst <= initq and not initd;

   GEN_ASYNC: if ASYNC_REGCLK_G=true generate
     U_SynchFifo : entity work.SynchronizerFifo
       generic map (TPD_G=>TPD_G,
                    DATA_WIDTH_G => 32,
                    ADDR_WIDTH_G => 2)
       port map (rst                => fifoRst,
                 wr_clk             => txclk,
                 wr_en              => '1',
                 din(14 downto 0)   => nToAvg,
                 din(15)            => '0',
                 din(31 downto 16)  => avgToWr,
                 rd_clk             => sysclk,
                 rd_en              => '1',
                 valid              => open,
                 dout(15 downto 0)  => nToAvgOut,
                 dout(31 downto 16) => avgToWrOut);
   end generate GEN_ASYNC;

   GEN_SYNC: if ASYNC_REGCLK_G=false generate
     nToAvgOut  <= '0' & nToAvg;
     avgToWrOUt <= avgToWr;
   end generate GEN_SYNC;
   
   process (txclk, txrst)
   begin  -- process
      if txrst = '1' then
         bsaInit    <= '0';
         bsaActive  <= '0';
         bsaAvgDone <= '0';
         bsaDone    <= '0';
         initd      <= '0';
         done       <= '1';
         nToAvg     <= (others=>'0');
         avgToWr    <= x"0000";
      elsif rising_edge(txclk) then
         if enable = '1' then
            bsaInit    <= initn after TPD_G;
            bsaActive  <= active after TPD_G;
            bsaAvgDone <= avgDone after TPD_G;
            bsaDone    <= donen and not done after TPD_G;  -- must overlap with bsaAvgDone
            initd      <= initq after TPD_G;
            done       <= donen after TPD_G;
            nToAvg     <= nToAvgn after TPD_G;
            avgToWr    <= avgToWrn after TPD_G;
         end if;
      end if;
   end process;

end BsaControl;

