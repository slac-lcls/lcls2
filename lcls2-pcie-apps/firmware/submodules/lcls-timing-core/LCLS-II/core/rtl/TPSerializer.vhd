-------------------------------------------------------------------------------
-- Title      : TPSerializer
-------------------------------------------------------------------------------
-- File       : TPSerializer.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-15
-- Last update: 2018-02-15
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Generates a 16b serial stream of the LCLS-II timing message.
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
use work.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use work.TPGPkg.all;
use work.StdRtlPkg.all;
use work.TimingPkg.all;
use work.CrcPkg.all;

entity TPSerializer is
   generic (
      TPD_G : time := 1 ns;
      Id : integer := 0
      );
   port (
      -- Clock and reset
      txClk      : in  sl;
      txRst      : in  sl;
      fiducial   : in  sl;
      msg        : in  TimingMessageType;
      advance    : in  sl;
      stream     : out TimingSerialType;
      streamId   : out slv(3 downto 0)
      );
end TPSerializer;

-- Define architecture for top level module
architecture TPSerializer of TPSerializer is

   constant NWORDS_C : slv(7 downto 0) := slv(conv_unsigned((TIMING_MESSAGE_BITS_C-1)/16,8));
   
   type RegType is record
      word_stream  : slv(TIMING_MESSAGE_BITS_C+15 downto 0);
      word_cnt     : slv(11 downto 0);
      ready        : sl;
   end record;
   constant REG_INIT_C : RegType := (
      word_stream  => (others => '0'),
      word_cnt     => (others => '0'),
      ready        => '0');

  signal r : RegType := REG_INIT_C;
  signal rin : RegType;
  
  attribute use_dsp48      : string;
  attribute use_dsp48 of r : signal is "yes";   
  
begin

  streamId      <= toSlv(Id,streamId'length);
  stream.ready  <= r.ready;
  stream.data   <= r.word_stream(15 downto 0);
  stream.offset <= (others=>'0');
  stream.last   <= '1';
  
  comb: process (r, msg, txRst, fiducial, advance)
    variable v    : RegType;
  begin 
      v := r;

      if fiducial='1' then
         --  Latch the timing frame into a shift register
         v.word_stream(TIMING_MESSAGE_BITS_C-1 downto 0)              := toSlv(msg);
         v.word_cnt := (others=>'0');
         v.ready    := '1';
      elsif advance='1' then
         --  Shift out the next word
         v.word_stream  := x"0000" & r.word_stream(r.word_stream'left downto 16);
         if (r.word_cnt=NWORDS_C) then
           v.ready := '0';
         else
           v.ready    := '1';
           v.word_cnt := r.word_cnt+1;
         end if;
      end if;

      if txRst='1' then
        v := REG_INIT_C;
      end if;
      
      rin <= v;

   end process;

   process (txClk)
   begin  -- process
      if rising_edge(txClk) then
         r <= rin after TPD_G;
      end if;
   end process;

end TPSerializer;
