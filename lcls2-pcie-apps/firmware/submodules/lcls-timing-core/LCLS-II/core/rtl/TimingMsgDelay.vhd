-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : TimingMsgDelay.vhd
-- Author     : 
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2014-05-02
-- Last update: 2016-01-11
-- Platform   : Vivado 2013.3
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
--
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
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

use work.StdRtlPkg.all;
use work.TimingPkg.all;


entity TimingMsgDelay is
   generic (
      -- General Configurations
      TPD_G             : time                        := 1 ns;
      BRAM_EN_G         : boolean                     := true;
      FIFO_ADDR_WIDTH_G : positive range 1 to (2**24) := 7);

   port (
      -- Timing Msg interface
      timingClk              : in  sl;
      timingRst              : in  sl;
      timingMessageIn        : in  TimingMessageType;
      timingMessageStrobeIn  : in  sl;
      delay                  : in  slv(15 downto 0);
      timingMessageOut       : out TimingMessageType;
      timingMessageStrobeOut : out sl);

end TimingMsgDelay;

architecture rtl of TimingMsgDelay is

   constant TIME_SIZE_C  : integer := 32;
   constant FIFO_WIDTH_C : integer := TIMING_MESSAGE_BITS_C + TIME_SIZE_C;

   subtype READOUT_RANGE_C is natural range TIMING_MESSAGE_BITS_C+TIME_SIZE_C-1 downto TIMING_MESSAGE_BITS_C;
   subtype TIMING_RANGE_C is natural range TIMING_MESSAGE_BITS_C-1 downto 0;

   type RegType is record
      timeNow                : slv(TIME_SIZE_C-1 downto 0);
      readoutTime            : slv(TIME_SIZE_C-1 downto 0);
      fifoRdEn               : sl;
      timingMessageOut       : TimingMessageType;
      timingMessageStrobeOut : sl;
   end record RegType;

   constant REG_INIT_C : RegType := (
      timeNow                => (others => '0'),
      readoutTime            => (others => '0'),
      fifoRdEn               => '0',
      timingMessageOut       => TIMING_MESSAGE_INIT_C,
      timingMessageStrobeOut => '0');

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal timingMessageSlv  : slv(TIMING_MESSAGE_BITS_C-1 downto 0);
   signal fifoTimingMessage : slv(TIMING_MESSAGE_BITS_C-1 downto 0);
   signal fifoReadoutTime   : slv(TIME_SIZE_C-1 downto 0);
   signal fifoValid         : sl;

begin

   timingMessageSlv <= toSlv(timingMessageIn);

   Fifo_Time : entity work.Fifo
      generic map (
         TPD_G           => TPD_G,
         GEN_SYNC_FIFO_G => false,
         BRAM_EN_G       => true,
         FWFT_EN_G       => true,
         USE_DSP48_G     => "no",
         USE_BUILT_IN_G  => false,
         DATA_WIDTH_G    => 32,
         ADDR_WIDTH_G    => 9)
      port map (
         rst    => timingRst,
         wr_clk => timingClk,
         wr_en  => timingMessageStrobeIn,
         din    => r.readoutTime,
         rd_clk => timingClk,
         rd_en  => r.fifoRdEn,
         dout   => fifoReadoutTime,
         valid  => fifoValid);

   Fifo_Data : entity work.Fifo
      generic map (
         TPD_G           => TPD_G,
         GEN_SYNC_FIFO_G => false,
         BRAM_EN_G       => true,
         FWFT_EN_G       => true,
         USE_DSP48_G     => "no",
         USE_BUILT_IN_G  => false,
         DATA_WIDTH_G    => TIMING_MESSAGE_BITS_C,
         ADDR_WIDTH_G    => 9)
      port map (
         rst    => timingRst,
         wr_clk => timingClk,
         wr_en  => timingMessageStrobeIn,
         din    => timingMessageSlv,
         rd_clk => timingClk,
         rd_en  => r.fifoRdEn,
         dout   => fifoTimingMessage,
         valid  => open);

   comb : process (delay, fifoReadoutTime, fifoTimingMessage, fifoValid, r, timingRst) is
      variable v : RegType;
   begin
      v := r;

      v.timeNow     := r.timeNow + 1;
      v.readoutTime := r.timeNow + delay;

      v.fifoRdEn               := '0';
      v.timingMessageStrobeOut := '0';
      v.timingMessageOut       := toTimingMessageType(fifoTimingMessage);

      if (fifoValid = '1' and r.fifoRdEn = '0') then
         if (fifoReadoutTime = r.timeNow) then
            v.fifoRdEn               := '1';
            v.timingMessageStrobeOut := '1';
         end if;
      end if;

      if (timingRst = '1') then
         v := REG_INIT_C;
      end if;

      rin <= v;

      timingMessageOut       <= r.timingMessageOut;
      timingMessageStrobeOut <= r.timingMessageStrobeOut;

   end process comb;

   seq : process (timingClk) is
   begin
      if (rising_edge(timingClk)) then
         r <= rin after TPD_G;
      end if;
   end process seq;

end rtl;
