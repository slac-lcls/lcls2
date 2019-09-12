-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : TimingFrameRx.vhd
-- Author     : Benjamin Reese  <bareese@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-01
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
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.TimingPkg.all;
use work.TimingExtnPkg.all;

entity TimingFrameRx is
   generic (
      TPD_G    : time    := 1 ns);
   port (
      rxClk               : in  sl;
      rxRst               : in  sl;
      rxData              : in  TimingRxType;

      messageDelay        : in  slv(19 downto 0);
      messageDelayRst     : in  sl;
      
      timingMessage       : out TimingMessageType;
      timingMessageStrobe : out sl;
      timingMessageValid  : out sl;
      timingExtn          : out TimingExtnType;
      timingExtnValid     : out sl;

      rxVersion           : out slv(31 downto 0);
      staData             : out slv(4 downto 0)
      );
end entity TimingFrameRx;

architecture rtl of TimingFrameRx is

   -------------------------------------------------------------------------------------------------
   -- rxClk Domain
   -------------------------------------------------------------------------------------------------
   type StateType is (IDLE_S, FRAME_S);

   type RegType is record
      vsnErr  : sl;
      version : slv(31 downto 0);
   end record;

   constant REG_INIT_C : RegType := (
     vsnErr  => '0',
     version => (others=>'1') );

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal fiducial           : sl;
   signal streams            : TimingSerialArray(1 downto 0);
   signal streamIds          : Slv4Array        (1 downto 0) := ( x"1", x"0" );
   signal advance            : slv              (1 downto 0);
   signal sof, eof, crcErr   : sl;
   signal dframe0            : slv(TIMING_MESSAGE_BITS_C-1 downto 0);
   signal dvalid0            : sl;
   signal doverflow0         : sl;
   signal dframe1            : slv(TIMING_EXTN_BITS_C-1 downto 0);
   signal dvalid1            : sl;
   signal dstrobe            : sl;
   signal delayRst           : sl;
   signal dmsg               : TimingMessageType;

begin

   delayRst <= rxRst or messageDelayRst;
   
   U_Deserializer : entity work.TimingDeserializer
      generic map ( TPD_G=>TPD_G, STREAMS_C => 2 )
      port map ( clk       => rxClk,
                 rst       => rxRst,
                 fiducial  => fiducial,
                 streams   => streams,
                 streamIds => streamIds,
                 advance   => advance,
                 data      => rxData,
                 sof       => sof,
                 eof       => eof,
                 crcErr    => crcErr );

   U_Delay0 : entity work.TimingSerialDelay
     generic map ( TPD_G=>TPD_G, NWORDS_G => TIMING_MESSAGE_WORDS_C,
                   FDEPTH_G => 100 )
     port map ( clk        => rxClk,
                rst        => delayRst,
                delay      => messageDelay,
                fiducial_i => fiducial,
                advance_i  => advance(0),
                stream_i   => streams(0),
                frame_o    => dframe0,
                strobe_o   => dstrobe,
                valid_o    => dvalid0,
                overflow_o => doverflow0);

   U_Extn : entity work.TimingSerialDelay
     generic map ( TPD_G=>TPD_G, NWORDS_G => TIMING_EXTN_BITS_C/16,
                   FDEPTH_G => 100 )
     port map ( clk        => rxClk,
                rst        => delayRst,
                delay      => messageDelay,
                fiducial_i => fiducial,
                advance_i  => advance(1),
                stream_i   => streams(1),
                frame_o    => dframe1,
                strobe_o   => open,
                valid_o    => dvalid1 );

   dmsg                <= toTimingMessageType(dframe0);
   timingMessage       <= dmsg;
   timingMessageStrobe <= dstrobe;
   timingMessageValid  <= dvalid0 and not r.vsnErr;
   timingExtn          <= toTimingExtnType(dframe1);
   timingExtnValid     <= dvalid1;
   rxVersion           <= r.version;
   staData             <= r.vsnErr & (crcErr or doverflow0) & fiducial & eof & sof;

   comb: process ( r, dmsg, dstrobe ) is
     variable v : RegType;
   begin
     v := r;

     if dstrobe = '1' then
       v.version := x"0000" & dmsg.version;
       if dmsg.version=TIMING_MESSAGE_VERSION_C then
         v.vsnErr := '0';
       else
         v.vsnErr := '1';
       end if;
     end if;

     rin <= v;
   end process;

   seq: process ( rxClk ) is
   begin
     if rising_edge(rxClk) then
       r <= rin;
     end if;
   end process;
     
end architecture rtl;

