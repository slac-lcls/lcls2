-------------------------------------------------------------------------------
-- Title         : EvrV2FromV1
-- Project       : LCLS-II Timing Pattern Generator
-------------------------------------------------------------------------------
-- File          : EvrV2FromV1.vhd
-- Author        : Matt Weaver, weaver@slac.stanford.edu
-- Created       : 01/23/2016
-------------------------------------------------------------------------------
-- Description:
-- Reformats LCLS timing stream into LCLS2 timing frame.  Only data buffer and
-- sequencer eventcodes are reformatted.
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
-- 01/23/2016: created.
-------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

use work.all;
use work.StdRtlPkg.all;
use work.TimingPkg.all;
use work.EvrV2Pkg.all;

entity EvrV2FromV1 is
  generic ( TPD_G      : time := 1 ns );
  port ( clk        : in  sl;
         disable    : in  sl;
         timingIn   : in  TimingBusType;
         timingOut  : out TimingMessageType );
end EvrV2FromV1;

architecture EvrV2FromV1 of EvrV2FromV1 is
  
begin

  process (clk)
    variable acrate : integer;
    variable destn  : integer;
  begin  -- process
    if rising_edge(clk) then
      if timingIn.strobe='1' then
        if disable = '0' then  -- map LCLS timing stream to look like LCLS-II frame
          timingOut.pulseId     <= resize(timingIn.stream.pulseId,17) &
                                   timingIn.stream.dbuff.epicsTime(31 downto 17) &
                                   timingIn.stream.dbuff.epicsTime(63 downto 32);
          timingOut.timeStamp   <= timingIn.stream.dbuff.epicsTime(31 downto 0) &
                                   timingIn.stream.dbuff.epicsTime(63 downto 32);
          timingOut.bsaInit     <= resize(timingIn.stream.dbuff.edefInit,64);
          -- encode minor/major mask on bsaInit
          timingOut.bsaActive   <= (resize(timingIn.stream.dbuff.edefMinor,64) and
                                    resize(timingIn.stream.dbuff.edefInit,64)) or
                                   (resize(timingIn.stream.dbuff.dmod(147 downto 128),64) and not
                                    resize(timingIn.stream.dbuff.edefInit,64));
          timingOut.bsaAvgDone  <= (resize(timingIn.stream.dbuff.edefMajor,64) and
                                    resize(timingIn.stream.dbuff.edefInit,64)) or
                                   (resize(timingIn.stream.dbuff.edefAvgDn,64) and not
                                    resize(timingIn.stream.dbuff.edefInit,64));
          timingOut.bsaDone     <= resize(timingIn.stream.dbuff.edefAvgDn,64);
          timingOut.fixedRates  <= (others=>'0');
          timingOut.acRates(0)  <= '1';
          timingOut.acRates(5 downto 1) <= timingIn.stream.dbuff.dmod(152 downto 148);
          timingOut.acTimeSlot <= timingIn.stream.dbuff.dmod(127 downto 125);
          --  Map all 6 modifier words
          timingOut.beamEnergy(0)    <= timingIn.stream.dbuff.dmod( 15 downto   0);
          timingOut.beamEnergy(1)    <= timingIn.stream.dbuff.dmod( 31 downto  16);
          timingOut.beamEnergy(2)    <= timingIn.stream.dbuff.dmod( 47 downto  32);
          timingOut.beamEnergy(3)    <= timingIn.stream.dbuff.dmod( 63 downto  48);
          timingOut.photonWavelen(0) <= timingIn.stream.dbuff.dmod( 79 downto  64);
          timingOut.photonWavelen(1) <= timingIn.stream.dbuff.dmod( 95 downto  80);
          timingOut.control(16)      <= timingIn.stream.dbuff.dmod(111 downto  96);
          timingOut.mpsLimit         <= timingIn.stream.dbuff.dmod(127 downto 112);
          for i in 0 to 15 loop
            timingOut.mpsClass(i)    <= timingIn.stream.dbuff.dmod(131+4*i downto 128+4*i);
          end loop;
          for i in 0 to 15 loop
            timingOut.control(i) <= timingIn.stream.eventCodes(i*16+15 downto i*16);
          end loop;
          timingOut.control(17) <= (others=>'0');
          -- Simulate beam request word : charge=0, dest={D10DMP,LI25,UND}, beam=POCKCEL
          destn := 2;
          if timingIn.stream.dbuff.dmod(61)='1' then
            destn := 0;
          end if;
          if timingIn.stream.dbuff.dmod(60)='1' then
            destn := 1;
          end if;
          timingOut.beamRequest <= x"000000" & toSlv(destn,4) & "000" & timingIn.stream.dbuff.dmod(83);
          --
        else
          timingOut <= timingIn.message;
        end if;
      end if;
    end if;
  end process;
  
end EvrV2FromV1;
